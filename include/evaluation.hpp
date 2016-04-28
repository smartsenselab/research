#ifndef EVALUATION
#define EVALUATION

#define _SCL_SECURE_NO_WARNINGS
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <functional>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <map>

namespace vr
{
	class Evaluation
	{
	public:

		std::string inputFile;
		std::vector<std::string> labelsName;
		cv::Mat_<float> confusionMat;
		std::vector<float> **confusionMatScores;

		Evaluation(std::string inputFile, std::string indexFile);
		~Evaluation();
		void evaluation();

	private:

		inline void generateOutput();
		double meanAccuracy(cv::Mat_<float> list);
		double stdDeviation(cv::Mat_<float> list, double mean);
		float balancedAccuracy(int TP, int FP, int FN, int TN);
		std::vector<std::string> readIndexFile(std::string indexFile);
		std::vector<std::string> splitString(std::string str, char delimiter);
		float averagePrecision(int label, int numTp, int numFn, std::map<int, std::vector<float>> TPScores, std::map<int, std::vector<float>> FPScores);

	};
}
#endif