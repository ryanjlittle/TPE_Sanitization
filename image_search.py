"""
Code copied from Adrian Rosebrock, https://pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
"""

import numpy as np
import csv
import argparse
import cv2
from index_dataset import ColorDescriptor

class Searcher:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def search(self, queryFeatures, limit = 10):
        # initialize our dictionary of results
        results = {}
        # open the index file for reading
        with open(self.indexPath) as f:
            # initialize the CSV reader
            reader = csv.reader(f)
            # loop over the rows in the index
            ct = 1
            for row in reader:
                ct += 1
                # parse out the image ID and features, then compute the
                # chi-squared distance between the features in our index
                # and our query features
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures)
                # now that we have the distance between the two feature
                # vectors, we can udpate the results dictionary -- the
                # key is the current image ID in the index and the
                # value is the distance we just computed, representing
                # how 'similar' the image in the index is to our query
                results[row[0]] = d
            # close the reader
            f.close()
        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])
        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d

if __name__=="__main__":
# construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
    ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
    ap.add_argument("-r", "--result-path", required = True, help = "Path to the result path")
    args = vars(ap.parse_args())
    # initialize the image descriptor
    cd = ColorDescriptor((8, 12, 3))
    # load the query image and describe it
    print(args["query"])
    query = cv2.imread(args["query"])
    cv2.imshow("Query", query)
    cv2.waitKey(0)
    
    features = cd.describe(query)
    # perform the search
    searcher = Searcher(args["index"])
    results = searcher.search(features)
    # display the query
    cv2.imshow("Query", query)
    # loop over the results
    for (score, resultID) in results:
        # load the result image and display it
        result = cv2.imread(args["result_path"] + "/" + resultID)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
