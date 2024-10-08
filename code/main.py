import skimage.io as skio

from utils import morph, mean_face_driver, caricature, morph_into

# # Parts 1-3
# morph(['ethan', 'shai'])
# morph(['tzuyu', 'minjoo'])
# morph(['lia', 'julie'])
# morph(['ethan_young', 'ethan_now'])
# morph(['ethan_kuo', 'ehong_kuo', 'dad_kuo', 'mom_kuo', 'ejean_kuo'])

# # Part 4
# ethan_face = skio.imread(f'../data/ethan_grayscale.jpg')
# mean_face_driver('a', 'neutral', ethan_face, 'ethan_grayscale', new_mean_face=True)
# mean_face_driver('b', 'happy', ethan_face, 'ethan_grayscale', new_mean_face=True)

# # Part 5
ethan_face = skio.imread(f'../data/ethan_grayscale.jpg')
neutral_face = skio.imread(f'../images/neutral.jpg')
caricature('ethan_grayscale', ethan_face, 'neutral', neutral_face)
caricature('ethan_grayscale', ethan_face, 'neutral', neutral_face, alpha=0)
caricature('ethan_grayscale', ethan_face, 'neutral', neutral_face, alpha=0.5)
caricature('ethan_grayscale', ethan_face, 'neutral', neutral_face, alpha=1)
caricature('ethan_grayscale', ethan_face, 'neutral', neutral_face, alpha=1.5)
caricature('ethan_grayscale', ethan_face, 'neutral', neutral_face, alpha=2)