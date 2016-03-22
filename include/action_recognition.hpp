#ifndef ACTION_RECOGNITION
#define ACTION_RECOGNITION


#include <string>
#include <iomanip>
#include <thread>
#include <numeric>
#include <opencv2\opencv.hpp>

#include "descriptors/descriptor_temporal.hpp"
#include "descriptors/ofcm_features.hpp"
#include "visual_dictionary.hpp"
#include "svm_multiclass.hpp"

#include <dirent.h>

namespace ccr
{
////// for dirent on windows //////
#define	isFile 32768
#define isDir  16384
///////////////////////////////////

	enum ClassificationProtocol
	{
		Train,
		Test,
		LeaveOneOut
	};

	struct SamplingSetup
	{
		int sampleX, sampleY, sampleL;
		int strideX, strideY,	strideL;
	};

	class ActionRecognition {
	private:
		ClassificationProtocol classificationProtocol;
		
		//long numExtractFeatures;
		std::string videosYMLPath;
		std::string outputFile;
		SamplingSetup samplingSetup;
		std::vector<cv::Mat> video;
		std::vector<ssig::Cube> cuboids;
		std::vector<FeatureIndex> featuresProperties;
		std::map<std::string, std::vector<cv::Mat>> mapLabelToBoW;
		ssig::DescriptorTemporal *desc;
		ccr::VisualDictionary *dict;
		ccr::Classification *classifier;

	public:

		cv::FileStorage params;

		ActionRecognition(std::string paramsPath);
		~ActionRecognition();
		void execute();
		void beforeProcess();
		void extractFeatures();
		void createDictionary();
		void extractBagOfWords();
		void learnClassificationModel();
		void classification();
		void loadDictionary();
		void loadClassifierModel();
		void createBoW(cv::Mat bagOfWords, std::string featurePath);

		void loadVideoFrames(cv::FileNodeIterator &inode);
		void createCuboids();
		void saveFeature(std::string label, cv::Mat &features, std::string videoName);
		std::vector<std::string> retrieveClassIds();
		void clearNoLongerUseful();

		inline void fillFeaturesProperties();
		inline void generateOutput(int nLabels, cv::Mat_<float> confusionMat, std::vector<float> **confusionMatScores);

	};

	////// for dirent on windows //////
	//static int isFile = 32768;
	//static int isDir = 16384;
	///////////////////////////////////

	std::vector<std::string> splitString(std::string str, char delimiter);
	std::vector<int> splitTemporalScales(std::string str, char delimiter);
	std::string getFileName(std::string videoName, cv::FileStorage &params);
	double meanAccuracy(cv::Mat_<float> list);
	double stdDeviation(cv::Mat_<float> list, double mean);
	float averagePrecision(int label, int numTp, int numFn, std::map<int, std::vector<float>> TPScores, std::map<int, std::vector<float>> FPScores);
}

#endif