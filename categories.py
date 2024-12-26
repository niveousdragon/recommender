
def process_string(s):
    return s.split('(')[0].strip()

def get_categories():
    with open('categories.txt', 'r+', encoding='utf-8') as f:
        content = f.read()

    strings = content.split('\n')

    all_cats = {}
    for s in strings:
        if s[:2] == '--':
            ccat = s[2:]
            all_cats.update({ccat:[]})

        elif len(s) == 0:
            pass

        else:
            all_cats[ccat].append(process_string(s))

    return all_cats
    #print(all_cats)
