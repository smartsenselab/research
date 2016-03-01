#ifndef VISUAL_DICTIONARY
#define VISUAL_DICTIONARY

#include <string>
#include <opencv2\opencv.hpp>

//Carlos Caetano Research
namespace ccr
{
	enum VisualDictionaryMethod
	{
		Random,
		Kmeans
	};

	class  VisualDictionary {

	private:
		int nCWs;					// Number of Codewords
		VisualDictionaryMethod method;
		cv::Mat_<float> dictionary;
		cv::Mat_<float> data;
		std::vector<std::string> labels;	/* optional labels associated with each feature vector */

		// Return de distance between two Matrices
		float distance(cv::Mat_<float> dict, cv::Mat_<float> data);

	public:

		// construtor
		VisualDictionary();
		VisualDictionary(int nCWs, VisualDictionaryMethod method);

		// destrutor
		~VisualDictionary();

		// Local setup
		void beforeProcess();

		// Add feature vectors
		void addFeatureVectors(const cv::Mat_<float> &data);

		// Add feature vectors with one label per vector
		void addFeatureVectors(const cv::Mat_<float> &data, const std::vector<std::string> &labels);

		// Function to return the dictionary
		void buildDictionary();

		// Compute bag
		void computeBag(cv::Mat_<float> &data, cv::Mat_<float> &bag);

		// Save dictionary
		void save(cv::FileStorage &storage);

		// Load dictionary
		void load(const cv::FileNode &node, cv::FileStorage &storage);

	};
}
#endif
