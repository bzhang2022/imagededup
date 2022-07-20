import glob
from imagededup.methods import PHash

if __name__ == "__main__":
    phasher = PHash()

    encodings = phasher.encode_images(image_dir='./imgs/Sinney')
    duplicates = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=10, scores=False, outfile='hash_results.json')

    # count = 0
    # encodings = {}
    # for img_name in glob.glob("./imgs/Sinney/*.jpg"):
    #     encoding = phasher.encode_image(image_file=img_name)
    #     encodings[img_name.split('\\')[-1]] = encoding
    #     count += 1
    # duplicates = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=4, scores=False, outfile='hash_results.json')
