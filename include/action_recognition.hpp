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

	enum ClassificationType
	{
		OneAgainstOne,						//OAO
		OneAgainstAll,						//OAA, seleciona a respota com maior Score
		OneAgainstAllMultiLabel		//OAA, permite vídeo ter mais de uma label
	};

	struct SamplingSetup
	{
		int sampleX, sampleY, sampleL;
		int strideX, strideY,	strideL;
	};

	struct ClassifierResponse {
		int i;
		int realClass;
		int predictedClass;
		float resp;
	};

	class ActionRecognition {
	private:
		ClassificationProtocol classificationProtocol;
		ClassificationType classificationType;
		
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
		std::vector<ccr::Classification*> classifiers;
		//ccr::Classification *classifier;
		int saveBinaryFile = 0;

	public:

		cv::FileStorage params;

		ActionRecognition(std::string paramsPath);
		~ActionRecognition();
		void execute();
		void beforeProcess();
		void extractFeatures();
		void createDictionary();
		void createDictionary2();
		void extractBagOfWords();
		void learnClassificationModel();
		void learnOneAgainstOneClassification();
		void learnOneAgainstAllClassification();
		void classification();
		void classificationThread(ClassifierResponse *cr);
		void deleteClassifiers();

		void oneAgainstOneClassification();
		void oneAgainstAllClassification();
		void oneAgainstAllMultiLabelClassification();

		void leaveOneOut();
		void loadDictionary();
		void loadClassifierModel();
		void loadOneAgainstOneClassifierModel();
		void loadOneAgainstAllClassifierModel();
		void createBoW(cv::Mat bagOfWords, std::string featurePath);

		void loadVideoFrames(cv::FileNodeIterator &inode);
		void createCuboids();
		void saveFeature(std::string label, cv::Mat &features, std::string videoName);
		void saveFeatureBinary(std::string label, cv::Mat &features, std::string videoName);
		void saveBoW(std::string label, cv::Mat &features, std::string videoName);
		std::vector<std::string> retrieveClassIds();
		std::vector<std::string> retrieveLabelsNameOneAgainstAll();
		void clearNoLongerUseful();
		cv::Mat matRead(const std::string& filename, std::string &label);
		std::string getFileName(std::string videoName, cv::FileStorage &params);

		inline void fillFeaturesProperties();
		inline void generateOutput(int nLabels, cv::Mat_<float> confusionMat, std::vector<float> **confusionMatScores);
		inline void generateOutputMultiLabel(std::vector<cv::Mat_<float>> vecConfusionMat, std::vector<std::vector<float>**> vecConfusionMatScores);
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