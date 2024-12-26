from cgitb import enable

from pymystem3 import Mystem
from categories import get_categories
STOP_LIST = ['и', 'для', 'все', 'всё', 'на', 'за', 'товары', 'товар', 'с', 'изделия', 'оплата']
mystem = Mystem()

def preprocess_group(c, enable_stemming=False):
    prepr = [w.lower().split('-')[0].replace(':', '').replace(',', '').replace("'", '').replace('[', '').replace(']', '') for w in c.split(' ')]
    if enable_stemming:
        meaning = tuple(mystem.lemmatize(p)[0] for p in prepr if p not in STOP_LIST)
    else:
        meaning = tuple(p for p in prepr if p not in STOP_LIST)

    return meaning


def get_all_preprocessed_subcategories(enable_stemming=False):
    all_cats = get_categories()
    # preprocess subcategories
    all_subcats = []
    reverse_cat = {}
    for cat, subcats in all_cats.items():
        for c in subcats:
            #print(c)
            meaning = preprocess_group(c, enable_stemming=enable_stemming)
            #print(meaning)
            all_subcats.append(meaning)
            reverse_cat.update({meaning: cat})

    return all_subcats, reverse_cat

#get_all_preprocessed_subcategories()