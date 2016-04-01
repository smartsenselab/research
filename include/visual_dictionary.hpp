#ifndef VISUAL_DICTIONARY
#define VISUAL_DICTIONARY

#include <string>
#include <thread>
#include <random>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <dirent.h>

//Carlos Caetano Research
namespace ccr
{
	////// for dirent on windows //////
	#define	isFile 32768
	#define isDir  16384
	///////////////////////////////////

	enum VisualDictionaryMethod
	{
		Random,
		Kmeans
	};

	struct FeatureIndex
	{
		std::string path;
		std::string label;
		int rows;
		unsigned short cols;
		int index;

		FeatureIndex(std::string p, std::string l, int r, unsigned short c, int i)
		{
			path = p;
			label = l;
			rows = r;
			cols = c;
			index = i;
		}
	};

	class  VisualDictionary {

	private:
		int nCWs;					// Number of Codewords
		VisualDictionaryMethod method;
		cv::Mat_<float> dictionary;
		std::vector<FeatureIndex> data;
		std::vector<std::string> pathFiles;
		std::vector<std::string> labels;	/* optional labels associated with each feature vector */
		std::vector<FeatureIndex> featuresProperties;

		// Return de distance between two Matrices
		float distance(cv::Mat dict, cv::Mat data);

	public:

		// construtor
		VisualDictionary();
		VisualDictionary(int nCWs, VisualDictionaryMethod method);
		int getnCWs();
		std::vector<FeatureIndex> getFeaturesProperties();
		void clearFeaturesProperties();

		// destrutor
		~VisualDictionary();

		// Local setup
		void beforeProcess();

		// Add feature vectors
		void addFeatureVectors(FeatureIndex &p);

		// Add feature vectors with one label per vector
		void addFeatureVectors(std::vector<FeatureIndex> fi, std::vector<std::string> &labels);

		// Function to return the dictionary
		void buildDictionary();
		void buildDictionary2(std::string path);
		void copyCodeWord(int i, cv::Mat_<float> &cluster, std::vector<size_t> &index);
		void copyCodeWord2(int i, int pos, cv::Mat_<float> &cluster, std::vector<int> &index);

		// Compute bag
		void computeBag(cv::Mat &data, cv::Mat &bag);

		// Save dictionary
		void save(cv::FileStorage &storage);

		// Load dictionary
		void load(const cv::FileNode &node, cv::FileStorage &storage);

	};
}
#endif
