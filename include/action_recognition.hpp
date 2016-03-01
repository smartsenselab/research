#ifndef ACTION_RECOGNITION
#define ACTION_RECOGNITION

#include <string>
#include <iomanip>
#include <opencv2\opencv.hpp>

#include "descriptors/descriptor_temporal.hpp"
#include "descriptors/ofcm_features.hpp"

namespace ccr
{
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
		SamplingSetup samplingSetup;
		ssig::DescriptorTemporal *desc;

	public:

		cv::FileStorage params;

		ActionRecognition(std::string paramsPath);
		~ActionRecognition();
		void execute();
		void beforeProcess();
		void extractFeatures();

		void loadVideoFrames(cv::FileNodeIterator &inode);
		void createCuboids();
		void saveFeature(std::string label, cv::Mat &features, std::string videoName);

	};

	std::vector<std::string> splitString(std::string str, char delimiter);
	std::vector<int> splitTemporalScales(std::string str, char delimiter);
	std::string getFileName(std::string videoName, cv::FileStorage &params);
}

#endif