#include "evaluation.hpp"

namespace vr
{

	Evaluation::Evaluation(std::string inputFile, std::string indexFile)
	{
		this->inputFile = inputFile;
		this->labelsName = readIndexFile(indexFile);

		int nLabels = this->labelsName.size();
		this->confusionMat.create(nLabels, nLabels);
		this->confusionMat = 0;

		this->confusionMatScores = new std::vector<float>*[nLabels];
		for (int i = 0; i < nLabels; i++)
			this->confusionMatScores[i] = new std::vector<float>[nLabels];
	}

	Evaluation::~Evaluation()
	{
		int nLabels = this->labelsName.size();

		this->labelsName.clear();
		this->confusionMat.release();

		if (this->confusionMatScores != NULL)
		{
			for (int i = 0; i < nLabels; i++)
				delete[] this->confusionMatScores[i];
			delete[] this->confusionMatScores;
			this->confusionMatScores = NULL;
		}
	}

	void Evaluation::evaluation()
	{
		std::ifstream file(this->inputFile);
		if (file.is_open())
		{
			std::string line;
			float resp;
			int realClass, predictedClass;

			while (getline(file, line))
			{
				std::vector<std::string> split = splitString(line, ' ');
				predictedClass = atoi(split[1].c_str()) - 1; // Victor started them from 1
				realClass = atoi(split[2].c_str()) - 1; // Victor started them from 1
				resp = static_cast<float>(atof(split[3].c_str()));

				this->confusionMat[realClass][predictedClass]++;
				this->confusionMatScores[realClass][predictedClass].push_back(resp);
			}
			file.close();
			generateOutput();
		}
		else
			std::cerr << "Wrong input file." << std::endl;
	}

	std::vector<std::string> Evaluation::readIndexFile(std::string indexFile)
	{
		std::vector<std::string> labelsName;
		std::ifstream file(indexFile);

		if (file.is_open())
		{
			std::string line;
			while (getline(file, line))
			{
				std::vector<std::string> split = splitString(line, ' ');
				//int realClass = atoi(split[0].c_str()) - 1; // Victor started them from 1
				labelsName.push_back(split[1]);
			}
		}

		return labelsName;
	}

	std::vector<std::string> Evaluation::splitString(std::string str, char delimiter)
	{
		std::vector<std::string> internal;
		std::stringstream ss(str);
		std::string tok;

		while (getline(ss, tok, delimiter))
			internal.push_back(tok);

		return internal;
	}

	inline void Evaluation::generateOutput()
	{
		cv::FileStorage storage;
		cv::Mat_<float> output;
		int nLabels = this->labelsName.size();
		output.release();
		output.create(cv::Size(9, nLabels)); //4 = TP, FP, FN, TN, precision, recall, specificity, accuracy per class, balanced accuracy per class 
		output = 0;
		float precision, recall, specificity;
		std::map<int, std::vector<float>> TPScores, FPScores, FNScores, TNScores; //first = class, second = scores

		//Create output file
		storage.open("output.yml", cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ClassificationOutput" << "{";

		// TP - True Positive
		for (int tp = 0; tp < nLabels; tp++)
		{
			output[tp][0] = this->confusionMat[tp][tp];
			for (int score = 0; score < this->confusionMatScores[tp][tp].size(); score++)
				TPScores[tp].push_back(this->confusionMatScores[tp][tp].at(score));
		}

		// FP - False Positive
		for (int col = 0; col < nLabels; col++)
		{
			for (int row = 0; row < nLabels; row++)
			{
				output[col][1] += this->confusionMat[row][col];

				if (col != row)
					for (int score = 0; score < this->confusionMatScores[row][col].size(); score++)
						FPScores[col].push_back(this->confusionMatScores[row][col].at(score));
			}
			output[col][1] -= output[col][0];
		}

		// FN - False Negative
		for (int row = 0; row < nLabels; row++)
		{
			for (int col = 0; col < nLabels; col++)
			{
				output[row][2] += this->confusionMat[row][col];

				if (col != row)
					for (int score = 0; score < this->confusionMatScores[row][col].size(); score++)
						FNScores[row].push_back(this->confusionMatScores[row][col].at(score));
			}
			output[row][2] -= output[row][0];
		}

		// TN - True Negative
		cv::Scalar sum = cv::sum(this->confusionMat);
		for (int tn = 0; tn < nLabels; tn++)
			output[tn][3] = sum[0] - output[tn][0] - output[tn][1] - output[tn][2];

		for (int l = 0; l < nLabels; l++)
		{
			precision = output[l][0] / (output[l][0] + output[l][1]);
			output[l][4] = precision;
			recall = output[l][0] / (output[l][0] + output[l][2]);
			output[l][5] = recall;
			specificity = output[l][3] / (output[l][3] + output[l][1]);
			output[l][6] = specificity;
			output[l][7] = (output[l][0] + output[l][3]) / (output[l][0] + output[l][1] + output[l][2] + output[l][3]); //accuracy per class
			output[l][8] = balancedAccuracy(output[l][0], output[l][1], output[l][2], output[l][3]); //balanced accuracy per class
		}

		float *ap = new float[nLabels];
		for (int label = 0; label < nLabels; label++)
			ap[label] = averagePrecision(label, output[label][0], output[label][2], TPScores, FPScores);

		double meanAP = std::accumulate(ap, ap + nLabels, 0.0f);
		double meanACC = meanAccuracy(output.col(7));
		double stdDev = stdDeviation(output.col(7), meanACC);
		double meanBalACC = meanAccuracy(output.col(8));
		double stdDevBal = stdDeviation(output.col(8), meanBalACC);

		storage << "MeanBalancedAccuracy" << meanBalACC;
		storage << "StandardDeviationBalACC" << stdDevBal;
		//storage << "MeanAccuracy" << meanACC; // Removed because we are using balanced accuracy
		//storage << "StandardDeviationACC" << stdDev; // Removed because we are using balanced accuracy
		storage << "MeanAveragePrecision" << meanAP;

		for (int label = 0; label < nLabels; label++)
		{
			std::stringstream classLabel;
			classLabel << "Class" << label;
			storage << classLabel.str() << "{";
			storage << "Label" << this->labelsName[label];
			storage << "TP" << output[label][0];
			storage << "FP" << output[label][1];
			storage << "FN" << output[label][2];
			storage << "TN" << output[label][3];
			storage << "precision" << output[label][4];
			storage << "recall" << output[label][5];
			storage << "specificity" << output[label][6];
			//storage << "accPerClass" << output[label][7]; // Removed because we are using balanced accuracy
			storage << "accBalPerClass" << output[label][8];
			storage << "apPerClass" << ap[label];
			storage << "}";
		}
		storage.release();

		delete ap;
	}

	double Evaluation::stdDeviation(cv::Mat_<float> list, double mean)
	{
		double sum = 0.0;
		double standDev = 0.0;

		for (int i = 0; i < list.rows; i++)
			sum += (list[i][0] - mean)*(list[i][0] - mean);

		sum = (double)(sum / (list.rows - 1));
		standDev = sqrt(sum);

		return standDev;
	}

	double Evaluation::meanAccuracy(cv::Mat_<float> list)
	{
		double sum = 0.0;
		double mean = 0.0;

		for (int i = 0; i < list.rows; i++)
			sum += list[i][0];

		mean = (double)(sum / (list.rows));

		return mean;
	}

	float Evaluation::balancedAccuracy(int TP, int FP, int FN, int TN)
	{
		float percentTP, percentFP, percentFN, percentTN;
		
		if (TP == 0)
			percentTP = 0.0;
		else
			percentTP = static_cast<float>(TP * 100.0) / static_cast<float>(TP + FN);
		percentFN = 100.0 - percentTP;

		if (FP == 0)
			percentFP = 0.0;
		else
			percentFP = static_cast<float>(FP * 100.0) / static_cast<float>(TN + FP);
		percentTN = 100.0 - percentFP;

		return (percentTP + percentTN) / (percentTP + percentTN + percentFN + percentFP); // Since we used percent values, it is the same as (sensitivity + specificity) / 2
	}

	float Evaluation::averagePrecision(int label, int numTp, int numFn, std::map<int, std::vector<float>> TPScores, std::map<int, std::vector<float>> FPScores)
	{
		std::vector<int> tpVec, fpVec;

		std::sort(TPScores[label].begin(), TPScores[label].end(), std::greater<float>());
		std::sort(FPScores[label].begin(), FPScores[label].end(), std::greater<float>());

		int i, j;
		i = j = 0;
		while (true)
		{
			if (TPScores[label].size() > 0 && FPScores[label].size() > 0)
			{
				if (TPScores[label][i] > FPScores[label][j])
				{
					tpVec.push_back(1);
					fpVec.push_back(0);
					i++;
				}
				else //(TPScores[label][i] < FPScores[label][j])
				{
					tpVec.push_back(0);
					fpVec.push_back(1);
					j++;
				}
			}
			if (i == TPScores[label].size() && j < FPScores[label].size()) //No more TPScore
			{
				while (j < FPScores[label].size())
				{
					tpVec.push_back(0);
					fpVec.push_back(1);
					j++;
				}
				break;
			}
			if (j == FPScores[label].size() && i < TPScores[label].size()) //No more TPScore
			{
				while (i < TPScores[label].size())
				{
					tpVec.push_back(1);
					fpVec.push_back(0);
					i++;
				}
				break;
			}
			if (FPScores[label].size() == 0 && TPScores[label].size() == 0)
				return 0.0;
		}

		int size = tpVec.size();

		int *tpCumsum = new int[size];
		std::partial_sum(tpVec.begin(), tpVec.end(), tpCumsum);

		int *fpCumsum = new int[size];
		std::partial_sum(fpVec.begin(), fpVec.end(), fpCumsum);

		float *precision = new float[size];
		for (int i = 0; i < size; i++)
			precision[i] = tpCumsum[i] / (float)(tpCumsum[i] + fpCumsum[i]);

		float *recall = new float[size];
		for (int i = 0; i < size; i++)
			recall[i] = tpCumsum[i] / (float)(numTp + numFn);

		float ap = 0.0;
		for (float t = 0.0; t <= 1; t += 0.1)
		{
			float max = -1.0;
			for (int i = 0; i < size; i++)
				if (recall[i] >= t)
					if (precision[i] >= max)
						max = precision[i];

			if (max == -1.0)
				max = 0.0;

			ap = ap + max / 11.0;
		}

		delete tpCumsum;
		delete fpCumsum;
		delete precision;
		delete recall;

		return ap;
	}

}