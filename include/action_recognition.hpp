#ifndef ACTION_RECOGNITION
#define ACTION_RECOGNITION


#include <string>
#include <iomanip>
#include <opencv2\opencv.hpp>

#include "descriptors/descriptor_temporal.hpp"
#include "descriptors/ofcm_features.hpp"
#include "visual_dictionary.hpp"

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
		TrainTest,
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
		
		long numExtractFeatures;
		std::string videosYMLPath;
		std::vector<cv::Mat> video;
		std::vector<ssig::Cube> cuboids;
		std::vector<std::string> featuresPath;
		SamplingSetup samplingSetup;
		ssig::DescriptorTemporal *desc;
		ccr::VisualDictionary *dict;

	public:

		cv::FileStorage params;

		ActionRecognition(std::string paramsPath);
		~ActionRecognition();
		void execute();
		void beforeProcess();
		void extractFeatures();
		void createDictionary();

		void loadVideoFrames(cv::FileNodeIterator &inode);
		void createCuboids();
		void saveFeature(std::string label, cv::Mat &features, std::string videoName);

		inline void addDictFeaturesPathFromFeatOutput();

	};

	////// for dirent on windows //////
	//static int isFile = 32768;
	//static int isDir = 16384;
	///////////////////////////////////

	std::vector<std::string> splitString(std::string str, char delimiter);
	std::vector<int> splitTemporalScales(std::string str, char delimiter);
	std::string getFileName(std::string videoName, cv::FileStorage &params);
}

#endif