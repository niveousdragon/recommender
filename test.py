from navec import Navec

npath = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(npath)

print(navec.sim('трусы', "белье"))