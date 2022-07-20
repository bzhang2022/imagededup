import glob
from imagededup.methods import CNN

if __name__ == "__main__":
    encoder = CNN()

    encodings = encoder.encode_images(image_dir='./imgs/Sinney')
    duplicates = encoder.find_duplicates(encoding_map=encodings, min_similarity_threshold=0.90, scores=False, outfile='cnn_results.json')

    # count = 0
    # encodings = {}
    # for img_name in glob.glob("./imgs/Sinney/*.jpg"):
    #     encoding = encoder.encode_image(image_file=img_name)
    #     encodings[img_name.split('/')[-1]] = encoding[0]
    #     count += 1
    # duplicates = encoder.find_duplicates(encoding_map=encodings, min_similarity_threshold=0.89, scores=False, outfile='cnn_results.json')
