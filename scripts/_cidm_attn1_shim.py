"""Uruchamia ICH `inference.py` bez zmiany, zdejmujac tylko `lora_id` z self-attencji.

Po co. Ich `inference.py` na diffusers 0.20.0 -- czyli na wersji, ktora sami przypinaja --
przewraca sie od razu na pierwszym kroku odszumiania (job 22581532, 22 s):

    TypeError: AttnProcessor2_0.__call__() got an unexpected keyword argument 'lora_id'

Przyczyna jest w ich kodzie i jest jednoznaczna. `revise_edlora_unet_fusionattention_forward`
(inference.py:123) podmienia procesor WYLACZNIE tam, gdzie w nazwie warstwy jest `attn2`:

    if layer.__class__.__name__ == 'Attention' and 'attn2' in name:

a `pipeline_edlora.py:296` wola UNet z `cross_attention_kwargs={'lora_id': id}`. W diffusers
0.20.0 `BasicTransformerBlock.forward` rozpakowuje te kwargs takze do `self.attn1`, gdzie
siedzi jeszcze fabryczny `AttnProcessor2_0` -- i ten nie zna `lora_id`.

Dlaczego zdjecie argumentu jest poprawne, a nie obejsciem. `attn1` to self-attencja i w ich
schemacie NIE MA na niej zadnej LoRA: `inference.py:157` zaklada `MultiLoRALinearLayer`
tylko na `attn2`. `lora_id` wybiera, ktory adapter ma zadzialac, wiec w warstwie bez zadnego
adaptera jest informacja bez odbiorcy. Fabryczny procesor liczy dokladnie to samo z nim i bez
niego.

Dlaczego nie edytujemy ich pliku. Wynik ma byc policzony ICH kodem -- to jest caly sens tego
wiersza w tabeli. Lata siedzi po naszej stronie, na fabrycznej klasie diffusers, i jest tego
samego rodzaju co zaslepki `_tkinter` i `IPython` w `sbatch_cidm_venv.sh`: ich zrodla zostaja
nietkniete, a my uzupelniamy to, czego brakuje w srodowisku.

Ich wlasny przelacznik `no_cross_attention_kwargs` (pipeline_edlora.py:215) tez by ten wyjatek
ominal, ale zmienia sciezke wywolania w attn2 i nie jest wystawiony w CLI, wiec zdjecie
martwego argumentu jest wezsza ingerencja.

Uzycie:  python scripts/_cidm_attn1_shim.py <argumenty ich inference.py>
         (uruchamiane z katalogu, w ktorym lezy ich inference.py)
"""
import runpy
import sys


def _drop_lora_id(cls):
    orig = cls.__call__

    def wrapped(self, *args, **kwargs):
        kwargs.pop("lora_id", None)
        return orig(self, *args, **kwargs)

    cls.__call__ = wrapped


def main():
    from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
    for cls in (AttnProcessor2_0, AttnProcessor):
        _drop_lora_id(cls)

    # Podmieniamy element ZEROWY, a nie zdejmujemy go: argparse czyta sys.argv[1:], bo
    # zaklada, ze [0] to nazwa programu. Zdjecie go przesuwalo wszystko o jeden i ich
    # parser gubil pierwsza flage, widzac jej wartosc jako argument pozycyjny
    # ("unrecognized arguments: _cfg_k49.json", job 3185173).
    sys.argv[0] = "inference.py"
    runpy.run_path("inference.py", run_name="__main__")


if __name__ == "__main__":
    main()
