
#ifndef CLASSIFICATION
#define CLASSIFICATION

#include <set>
#include <string>
#include <opencv2\opencv.hpp>

#define EXTRA_CLASS "classExtra"

namespace ccr
{
	// class to perform classification of feature vectors.
	class Classification {

	protected:
		int nsamples;					/* total number of samples */
		cv::Mat dataX;		/* data */
		std::vector<std::string> dataY;			/* labels */
		cv::FileStorage params;

		cv::Mat extraDataX;		/* data */
		std::vector<std::string> extraDataY;
		std::set<std::string> extraLabels;

		std::set<std::string> labels;				/* class labels */

		void save_(cv::FileStorage &storage);

		void load_(const cv::FileNode &node);

	public:
		Classification(cv::FileStorage &storage);
		virtual ~Classification();

		// load a classifier from a file
		void load(std::string filename);

		// save a classifier to a file
		void save(std::string filename);

		// retrieve class IDs (same order as the responses)
		virtual std::vector<std::string> retrieveClassIDs();

		// retrieve integer class IDs 0 means the first retrieved in RetrieveClassIDs(), and so one
		std::vector<int> retrieveClassIntIDs();

		// retrieve position in the response that will contain the response for a given class id
		int retrieveResponseClassIDPosition(std::string id);

		// Add training samples to the classifier.  it requires one classID per sample
		void addSamples(const cv::Mat &X, const std::vector<std::string> &Y);

		// Add training samples to the classifier. The id is set to all samples.
		void addSamples(const cv::Mat &X, std::string id);

		void addExtraSamples(const cv::Mat &X, const std::vector<std::string> &Y);

		void addExtraSamples(const cv::Mat &X);

		void addExtraSamples(const cv::Mat &X, std::string id);

		// reset classifier data
		void reset();

		virtual Classification *duplicateParameters() = 0;//To does not be must implement
		/*
		* Functions to be implemented by the specific methods
		*/
		// Classify samples and set responses.
		virtual void predict(const cv::Mat &X, cv::Mat &responses) = 0;

		// learn the classifier
		virtual void learn() = 0;

		// save classifier 
		virtual void save(cv::FileStorage &storage) = 0;

		// load classifier
		virtual void load(const cv::FileNode &node, cv::FileStorage &storage) = 0;
	};
}


#endif
