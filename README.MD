# Pastec feature extractor

This is a porting of the Pastec C++ code to python to visualize the features, using the same visualWords ORB descriptor file. The purpose is to see how we can replicated this data in another, more efficient storage and retrieval system.

## Usage

`python3 orb_feature_extraction.py https://iiif-bucket.s3.eu-west-1.amazonaws.com/digiteca/32044103326583/110485_1.jpg visualWordsORB.dat > keypoints.txt`

This image should have 1943 features according to the original pastec code. The output here confirms that.

## output file (keypoints.txt)

This shows both the keypoints and descriptors for a given image.

