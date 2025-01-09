from translate import Translator

translator = Translator(to_lang="ru")
translation = translator.translate("mangolin, vehicle oil, bumble")
print(translation)