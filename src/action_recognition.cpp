#include "action_recognition.hpp"

namespace ccr
{
	ActionRecognition::ActionRecognition(std::string paramsPath) {
		//numExtractFeatures = 0;
		this->params.open(paramsPath, cv::FileStorage::READ);
		beforeProcess();
	}

	ActionRecognition::~ActionRecognition() {
		clearNoLongerUseful();
		params.release();
		featuresProperties.clear();

		if (this->desc != NULL)
		{
			delete this->desc;
			desc = NULL;
		}
		if (this->dict != NULL)
		{
			delete this->dict;
			dict = NULL;
		}
		if (this->classifier != NULL)
		{
			delete this->classifier;
			classifier = NULL;
		}
	}

	void ActionRecognition::clearNoLongerUseful()
	{
		std::string rmdir, outputFold;

		rmdir = "RMDIR /S /Q ";
		params["featureOutput"] >> outputFold;
		system((rmdir + outputFold).c_str());
	}

	void ActionRecognition::beforeProcess() {
		int cP;
		cv::FileNode node;
		std::string mkdir, outputFold;
		params["videosFile"] >> videosYMLPath;
		params["outputFile"] >> outputFile;
		cP = static_cast<int>(params["classificationProtocol"]);
		
		switch (cP)
		{
		case ClassificationProtocol::Train:
			classificationProtocol = ClassificationProtocol::Train;
			break;
		case ClassificationProtocol::Test:
			classificationProtocol = ClassificationProtocol::Test;
			break;
		case ClassificationProtocol::LeaveOneOut:
			classificationProtocol = ClassificationProtocol::LeaveOneOut;
			break;
		default:
			classificationProtocol = ClassificationProtocol::Train;
		}


		mkdir = "MKDIR ";
		params["featureOutput"] >> outputFold;
		system((mkdir + outputFold).c_str());


		node = params["samplingSetup"]["blocks"];
		for (int i = 0; i < node.size(); ++i) {
			samplingSetup.sampleX = (int)node[i][0]; samplingSetup.sampleY = (int)node[i][1]; samplingSetup.sampleL = (int)node[i][2];
			samplingSetup.strideX = (int)node[i][3]; samplingSetup.strideY = (int)node[i][4]; samplingSetup.strideL = (int)node[i][5];
		}


		node = params["featureParams"];
		int nBMag = node["nBinsMagnitude"];
		int nBAng = node["nBinsAngle"];
		int distMag = node["distanceMagnitude"];
		int distAng = node["distanceAngle"];
		int cubL = node["cuboidLength"];
		float maxMag = node["maxMagnitude"];
		int logQ = node["logQuantization"];
		int movF = node["movementFilter"];
		std::string tempS; node["temporalScales"] >> tempS;
		std::vector<int> vec = splitTemporalScales(tempS, ',');

		desc = new ssig::OFCM(nBMag, nBAng, distMag, distAng, cubL, maxMag, logQ, static_cast<bool>(movF), vec);
		

		node = params["visualDictionayParams"];
		int nCWs = node["nCWs"];
		int method = node["method"];
		VisualDictionaryMethod vdMethod;

		switch (method)
		{
		case VisualDictionaryMethod::Random:
			vdMethod = VisualDictionaryMethod::Random;
			break;
		case VisualDictionaryMethod::Kmeans:
			vdMethod = VisualDictionaryMethod::Kmeans;
			break;
		default:
			vdMethod = VisualDictionaryMethod::Random;
		}

		dict = new ccr::VisualDictionary(nCWs, vdMethod);
		classifier = new ccr::SVM_Multiclass(params);
	}

	void ActionRecognition::execute()
	{

		switch (classificationProtocol)
		{
		case ClassificationProtocol::Train:
			extractFeatures();
			createDictionary();
			extractBagOfWords();
			learnClassificationModel();
			break;
			
		case ClassificationProtocol::Test:
			extractFeatures();
			loadDictionary();
			extractBagOfWords();
			loadClassifierModel();
			classification();
			break;
		/*
		case ClassificationProtocol::LeaveOneOut:
			extractFeatures();
			createDictionary();
			extractBagOfWords();
			leaveOneOut();
			break;
		*/
		}
	}

	void ActionRecognition::extractFeatures()
	{
		cv::Mat output;
		std::string videoName, label, outputFold, path, protocol = "";
		cv::FileNode node;
		cv::FileNodeIterator inode;
		cv::FileStorage videosStorage; // storageNumFeat;

		videosStorage.open(videosYMLPath, cv::FileStorage::READ);
		node = videosStorage["videos"];

		for (inode = node.begin(); inode != node.end(); ++inode) {
			videoName = (std::string) (*inode)["dir"];
			std::vector<std::string> split = splitString(videoName, '/');
			videoName = splitString(split[split.size() - 1], '.')[0];
			std::cout << "Extracting features from " << videoName;
			
			video.clear();
			
			loadVideoFrames(inode);
			label = (*inode)["objID"].isNone() ? "" : (std::string)(*inode)["objID"];
			std::cout << " .";

			cuboids.clear();
			createCuboids();
			std::cout << ".";

			output.release();
			desc->release();
			desc->setData(video);
			video.clear(); //not used anymore
			desc->extract(cuboids, output);
			std::cout << ".";
			//numExtractFeatures += output.rows;

			FeatureIndex fi(getFileName(videoName, params), label, output.rows, output.cols, 0); // 0 is  a dummy number
			featuresProperties.push_back(fi);

			desc->release();
			
			saveFeature(label, output, videoName);
			std::cout << " OK" << std::endl;
		}
		/*
		params["featureOutput"] >> outputFold;
		
		if (classificationProtocol == ClassificationProtocol::Train)
			protocol = "Train";
		else if (classificationProtocol == ClassificationProtocol::Test)
			protocol = "Test";

		path = outputFold + "\\numFeatures" + protocol + ".yml";
		storageNumFeat.open(path, cv::FileStorage::WRITE);
		if (storageNumFeat.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;
		storageNumFeat << "Features" << numExtractFeatures;

		storageNumFeat.release();
		*/
		output.release();
		videosStorage.release();
		video.clear();
		cuboids.clear();
		delete desc;
		desc = NULL;
	}

	void ActionRecognition::createDictionary()
	{
		cv::FileStorage storage;
		std::string dictModelFile;

		std::cout << "\n\nBuilding dictionary ";

		params["dictionaryFile"] >> dictModelFile;

		// Open modelFile
		storage.open(dictModelFile, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ActionRecognitionDictionary" << "{";

		std::cout << ".";

		if (featuresProperties.size() == 0)
			fillFeaturesProperties();

		for (auto &p : featuresProperties)
			dict->addFeatureVectors(p);

		std::cout << ".";

		// Create dictionary 
		dict->buildDictionary();
		std::cout << ".";

		// Save dictionary
		dict->save(storage);
		std::cout << " OK" << std::endl << std::endl;
		
		//featuresPath.clear();
		storage.release();
	}

	void ActionRecognition::extractBagOfWords()
	{
		int numVideos = 0;
		int v = 0;
		std::vector<std::thread> threads;

		if (featuresProperties.size() == 0)
			fillFeaturesProperties();

		numVideos = static_cast<int>(featuresProperties.size());
		
		while (v < numVideos)
		{
			std::cout << "Extracting Bag of Words for videos ";
			std::vector< std::pair<cv::Mat, std::string> > bagOfWordsVector;
			std::map<std::string, std::string> mapPathToLabel;
			for (int cores = 0; cores < std::thread::hardware_concurrency() && v < numVideos; cores++, v++)
			{
				cv::Mat bow;
				bow.release();
				bow.create(1, dict->getnCWs(), CV_32F);
				bow = 0;
				std::pair<cv::Mat, std::string> p(bow, featuresProperties[v].path);
				bagOfWordsVector.push_back(p);
				mapPathToLabel[p.second] = featuresProperties[v].label;
				std::cout << v << ", ";
			}
			std::cout << ".";
			for (auto& p : bagOfWordsVector)
				threads.push_back(std::thread(&ActionRecognition::createBoW, this, p.first, p.second));
			
			std::cout << ".";
			for (auto& th : threads)
				th.join();
			
			std::cout << ".";
			//for (auto& p : bagOfWordsVector) // Old way. Used just for one label per video
				//this->mapLabelToBoW[mapPathToLabel[p.second]].push_back(p.first); //Used just for one label per video

			for (auto& p : bagOfWordsVector) // New way. Used just for multi-label videos
			{
				std::vector<std::string> multiLabel = splitString(mapPathToLabel[p.second], ',');
				for (std::string &label : multiLabel)
					this->mapLabelToBoW[label].push_back(p.first);
			}

			std::cout << " OK" << std::endl;
			threads.clear();
		}
	}

	void ActionRecognition::createBoW(cv::Mat bagOfWords, std::string featurePath)
	{
		cv::Mat feature;
		cv::FileStorage storageFeat;
		cv::FileNode node, n1;

		//Loading feature
		storageFeat.open(featurePath, cv::FileStorage::READ);
		if (storageFeat.isOpened() == false)
			std::cerr << "Invalid file storage " << (featurePath + "!") << std::endl;

		node = storageFeat.root();
		n1 = node["ActionRecognitionFeatures"];
		n1["Features"] >> feature;
		storageFeat.release();

		for (int r = 0; r < feature.rows; r++) //hard assignment, ModObjectDetectionBOW parece fazer soft assignment...
		{
			cv::Mat partialBagging;
			dict->computeBag(feature.row(r), partialBagging);
			int idx = static_cast<int>(partialBagging.at<float>(1, 0)); //[0][0] contains the distance, [0][1] contains the index
			bagOfWords.at<float>(0, idx)+=1;
		}
		cv::normalize(bagOfWords, bagOfWords, 1, cv::NORM_L2);
	}

	void ActionRecognition::learnClassificationModel()
	{
		std::string classModelFile;
		cv::FileStorage storage;
		std::vector<std::string> labelsName;

		params["classModelFile"] >> classModelFile;

		// Open modelFile
		storage.open(classModelFile, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage " << (classModelFile + "!") << std::endl;

		storage << "ActionRecognitionClassificationModel" << "{";

		for (std::pair<std::string, std::vector<cv::Mat>> p : mapLabelToBoW)
			for (cv::Mat trainData : p.second)
				classifier->addSamples(trainData, p.first);

		classifier->learn();
		labelsName = classifier->retrieveClassIDs();
		storage << "Labels" << labelsName;
		classifier->save(storage);
		storage.release();
	}

	void ActionRecognition::loadDictionary()
	{
		cv::FileStorage storage;
		cv::FileNode node, n1;
		std::string dictModelFile;

		params["dictionaryFile"] >> dictModelFile;

		//Loading Dictionary
		storage.open(dictModelFile, cv::FileStorage::READ);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage " << dictModelFile << "!" << std::endl;

		node = storage.root();
		n1 = node["ActionRecognitionDictionary"];
		dict->load(n1, storage);
		storage.release();
	}

	void ActionRecognition::loadClassifierModel()
	{
		cv::FileStorage storage;
		cv::FileNode node, n1;
		std::string classModelFile;

		params["classModelFile"] >> classModelFile;

		//Loading Classifier Model
		storage.open(classModelFile, cv::FileStorage::READ);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage " << classModelFile << "!" << std::endl;

		node = storage.root();
		n1 = node["ActionRecognitionClassificationModel"];
		classifier->load(n1, storage);
		storage.release();
	}

	void ActionRecognition::createCuboids()
	{
		cuboids.clear();

		int videoHeight = video[0].rows;
		int videoWidth = video[0].cols;

		for (int t = 0; t <= static_cast<int>(0 + video.size() - samplingSetup.sampleL); t += samplingSetup.strideL)
			for (int y = 0; y <= static_cast<int>(0 + videoHeight - samplingSetup.sampleY); y += samplingSetup.strideY)
				for (int x = 0; x <= static_cast<int>(0 + videoWidth - samplingSetup.sampleX); x += samplingSetup.strideX)
					cuboids.push_back(ssig::Cube(x, y, t, samplingSetup.sampleX, samplingSetup.sampleY, samplingSetup.sampleL));
	}

	void ActionRecognition::loadVideoFrames(cv::FileNodeIterator &inode)
	{
		int step, nSamples = 0;
		unsigned long int videoLength;
		std::string path;
		cv::VideoCapture capture;
		cv::Mat image;
		
		video.clear();

		// load video
		path = (std::string) (*inode)["dir"];

		// check video loaded
		capture.open(path);
		if (!capture.isOpened())
			std::cerr << "Error processing file. Can't read video " << path << std::endl;

		videoLength = static_cast<unsigned long>(capture.get(CV_CAP_PROP_FRAME_COUNT));
		nSamples = (*inode)["nSamples"].isNone() ? 0 : (int)(*inode)["nSamples"];

		if (nSamples > 0)
			step = (int)(videoLength / nSamples); // generate step based on the number of samples
		else
			step = (*inode)["step"].isNone() ? 1 : (int)(*inode)["step"]; // load step
		
		for (unsigned long int frameStep = 0; frameStep < videoLength; frameStep += step)
		{
			capture.set(CV_CAP_PROP_POS_FRAMES, frameStep);
			capture.read((image));
			if (image.empty())
				std::cerr << "Error processing file. Can't read frame " << frameStep << "from video " << path << std::endl;
			video.push_back(image.clone());
		}

	}

	void ActionRecognition::saveFeature(std::string label, cv::Mat &features, std::string videoName) {
		
		cv::FileStorage storage;
		std::string path;
		// Open modelFile
		path = getFileName(videoName, params);
		
		storage.open(path, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ActionRecognitionFeatures" << "{";
		storage << "Label" << label;
		storage << "Features" << features;

		storage.release();
		
		//featuresPath.push_back(path);
	}

	inline void ActionRecognition::fillFeaturesProperties()
	{
		DIR *dir = 0;
		std::string path;
		int rows;
		unsigned short cols;
		std::string label;
		cv::FileNode node, n1;

		params["featureOutput"] >> path;
		dir = opendir(path.c_str());

		if (dir == 0) {
			std::cerr << "Impossible to open directory." << std::endl;
			exit(1);
		}

		struct dirent *featFile = 0;
		while (featFile = readdir(dir))
		{
			std::string file = featFile->d_name;
			//fazer verificação de AVI e JPG :-)
			if (featFile->d_type == isFile)
			{
				cv::FileStorage storageFeat;

				//Loading feature
				storageFeat.open(path + "\\" + file, cv::FileStorage::READ);
				if (storageFeat.isOpened() == false)
					std::cerr << "Invalid file storage " << (path + file + "!") << std::endl;

				node = storageFeat.root();
				n1 = node["ActionRecognitionFeatures"];
				n1["Label"] >> label;
				node = n1["Features"];
				node["rows"] >> rows;
				node["cols"] >> cols;
				storageFeat.release();

				FeatureIndex fi((path + "\\" + file), label, rows, cols, 0); // 0 is  a dummy number
				featuresProperties.push_back(fi);				
			}
		}
	}

	void ActionRecognition::classification()
	{
		int nLabels = classifier->retrieveClassIDs().size();
		cv::Mat_<float> confusionMat(nLabels, nLabels);
		std::vector<float> **confusionMatScores;
		confusionMat = 0;

		confusionMatScores = new std::vector<float>*[nLabels];
		for (int i = 0; i < nLabels; i++)
			confusionMatScores[i] = new std::vector<float>[nLabels];

		std::ofstream class_report;
		class_report.open("class_report.txt", std::ofstream::out | std::ofstream::app);
		class_report << "realClass\tpredictedClass\tresp" << std::endl;

		for (std::pair<std::string, std::vector<cv::Mat> > p : mapLabelToBoW)
		{
			int realClass = classifier->retrieveResponseClassIDPosition(p.first);
			for (cv::Mat_<float> testData : p.second)
			{
				cv::Mat_<float> responses;
				classifier->predict(testData, responses);
				int predictedClass = responses[0][0];
				float resp = responses[0][1];
				confusionMat[realClass][predictedClass]++;
				confusionMatScores[realClass][predictedClass].push_back(resp);

				class_report << realClass << "\t" << predictedClass << "\t" << resp << std::endl;
			}
		}

		class_report.close();

		generateOutput(nLabels, confusionMat, confusionMatScores);

		for (int i = 0; i < nLabels; i++)
			delete[] confusionMatScores[i];
		delete[] confusionMatScores;
	}

	inline void ActionRecognition::generateOutput(int nLabels, cv::Mat_<float> confusionMat, std::vector<float> **confusionMatScores)
	{
		cv::FileStorage storage;
		cv::Mat_<float> output;
		output.release();
		output.create(cv::Size(8, nLabels)); //4 = TP, FP, FN, TN, precision, recall, specificity, accuracy per class
		output = 0;
		float precision, recall, specificity;
		std::map<int, std::vector<float>> TPScores, FPScores, FNScores, TNScores; //first = class, second = scores

		//Create output file
		storage.open(outputFile, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!" << std::endl;

		storage << "ActionRecognitionClassificationOutput" << "{";

		// TP - True Positive
		for (int tp = 0; tp < nLabels; tp++)
		{
			output[tp][0] = confusionMat[tp][tp];
			for (int score = 0; score < confusionMatScores[tp][tp].size(); score++)
				TPScores[tp].push_back(confusionMatScores[tp][tp].at(score));
		}

		// FP - False Positive
		for (int col = 0; col < nLabels; col++)
		{
			for (int row = 0; row < nLabels; row++)
			{
				output[col][1] += confusionMat[row][col];

				if (col != row)
					for (int score = 0; score < confusionMatScores[row][col].size(); score++)
						FPScores[col].push_back(confusionMatScores[row][col].at(score));
			}
			output[col][1] -= output[col][0];
		}

		// FN - False Negative
		for (int row = 0; row < nLabels; row++)
		{
			for (int col = 0; col < nLabels; col++)
			{
				output[row][2] += confusionMat[row][col];

				if (col != row)
					for (int score = 0; score < confusionMatScores[row][col].size(); score++)
						FNScores[row].push_back(confusionMatScores[row][col].at(score));
			}
			output[row][2] -= output[row][0];
		}

		// TN - True Negative
		cv::Scalar sum = cv::sum(confusionMat);
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
		}

		float *ap = new float[nLabels];
		for (int label = 0; label < nLabels; label++)
			ap[label] = averagePrecision(label, output[label][0], output[label][2], TPScores, FPScores);

		double meanAP = std::accumulate(ap, ap + nLabels, 0.0f);
		double meanACC = meanAccuracy(output.col(7));
		double stdDev = stdDeviation(output.col(7), meanACC);

		storage << "MeanAccuracy" << meanACC;
		storage << "StandardDeviationACC" << stdDev;
		storage << "MeanAveragePrecision" << meanAP;

		std::vector<std::string> labelsName = retrieveClassIds();
		for (int label = 0; label < nLabels; label++)
		{
			std::stringstream classLabel;
			classLabel << "Class" << label;
			storage << classLabel.str() << "{";
			storage << "Label" << labelsName[label];
			storage << "TP" << output[label][0];
			storage << "FP" << output[label][1];
			storage << "FN" << output[label][2];
			storage << "TN" << output[label][3];
			storage << "precision" << output[label][4];
			storage << "recall" << output[label][5];
			storage << "specificity" << output[label][6];
			storage << "accPerClass" << output[label][7];
			storage << "apPerClass" << ap[label];
			storage << "}";
		}
		storage.release();

		delete ap;
	}

	std::vector<std::string> ActionRecognition::retrieveClassIds()
	{
		std::vector<std::string> ids;
		classifier->reset();

		for (auto& p : mapLabelToBoW)
			ids.push_back(p.first);

		return ids;
	}

	double stdDeviation(cv::Mat_<float> list, double mean)
	{
		double sum = 0.0;
		double standDev = 0.0;

		for (int i = 0; i < list.rows; i++)
			sum += (list[i][0] - mean)*(list[i][0] - mean);

		sum = (double)(sum / (list.rows - 1));
		standDev = sqrt(sum);

		return standDev;
	}

	double meanAccuracy(cv::Mat_<float> list)
	{
		double sum = 0.0;
		double mean = 0.0;

		for (int i = 0; i < list.rows; i++)
			sum += list[i][0];

		mean = (double)(sum / (list.rows));

		return mean;
	}

	float averagePrecision(int label, int numTp, int numFn, std::map<int, std::vector<float>> TPScores, std::map<int, std::vector<float>> FPScores)
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

	std::string getFileName(std::string videoName, cv::FileStorage &params)
	{
		short i = 1;
		bool fileExists = false;
		std::string outputFold, path;
		// Open modelFile
		params["featureOutput"] >> outputFold;
		//system((mkdir + outputFold).c_str());
		path = outputFold + "\\" + videoName + ".yml";

		do
		{
			if (FILE *file = fopen(path.c_str(), "r"))
			{
				char buffer[3];
				fileExists = true;
				itoa(i++, buffer, 10);
				path = outputFold + "\\" + videoName + "_" + buffer + ".yml";
				fclose(file);
			}
			else
				fileExists = false;

		} while (fileExists);
		
		return path;
	}

	std::vector<std::string> splitString(std::string str, char delimiter)
	{
		std::vector<std::string> internal;
		std::stringstream ss(str);
		std::string tok;

		while (getline(ss, tok, delimiter))
			internal.push_back(tok);

		return internal;
	}

	std::vector<int> splitTemporalScales(std::string str, char delimiter)
	{
		std::vector<int> internal;
		std::stringstream ss(str);
		std::string tok;

		while (getline(ss, tok, delimiter)) {
			int ts = atoi(tok.c_str());
			internal.push_back(ts);
		}

		return internal;
	}
}