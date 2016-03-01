#include "action_recognition.hpp"

namespace ccr
{
	ActionRecognition::ActionRecognition(std::string paramsPath) {
		numExtractFeatures = 0;
		this->params.open(paramsPath, cv::FileStorage::READ);
		beforeProcess();
	}

	ActionRecognition::~ActionRecognition() {
		params.release();

		if (this->desc != NULL)
		{
			delete this->desc;
			desc = NULL;
		}
	}

	void ActionRecognition::beforeProcess() {
		int cP;
		cv::FileNode node;
		std::string mkdir, outputFold;
		params["videosFile"] >> videosYMLPath;
		cP = static_cast<int>(params["classificationProtocol"]);
		
		switch (cP)
		{
		case ClassificationProtocol::Train:
			classificationProtocol = ClassificationProtocol::Train;
			break;
		case ClassificationProtocol::Test:
			classificationProtocol = ClassificationProtocol::Test;
			break;
		case ClassificationProtocol::TrainTest:
			classificationProtocol = ClassificationProtocol::TrainTest;
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
	}

	void ActionRecognition::extractFeatures()
	{
		cv::Mat output;
		std::string videoName, label, outputFold, path, protocol = "";
		cv::FileNode node;
		cv::FileNodeIterator inode;
		cv::FileStorage videosStorage, storageNumFeat;

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
			numExtractFeatures += output.rows;
			desc->release();

			saveFeature(label, output, videoName);
			std::cout << " OK" << std::endl;
		}

		params["featureOutput"] >> outputFold;
		
		if (classificationProtocol == ClassificationProtocol::Train)
			protocol = "Train";
		else if (classificationProtocol == ClassificationProtocol::Test)
			protocol = "Test";

		path = outputFold + "\\numFeatures" + protocol + ".yml";
		storageNumFeat.open(path, cv::FileStorage::WRITE);
		if (storageNumFeat.isOpened() == false)
			std::cerr << "Invalid file storage!";
		storageNumFeat << "Features" << numExtractFeatures;

		storageNumFeat.release();
		output.release();
		videosStorage.release();
		video.clear();
		cuboids.clear();
		delete desc;
		desc = NULL;
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
			std::cerr << "Error processing file. Can't read video " << path;

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
				std::cerr << "Error processing file. Can't read frame " << frameStep << "from video " << path;
			video.push_back(image.clone());
		}

	}

	void ActionRecognition::execute()
	{
		
		switch (classificationProtocol)
		{
		case ClassificationProtocol::Train:
			extractFeatures();
			break;
			/*
		case ClassificationProtocol::Test:
			
			break;
		case ClassificationProtocol::TrainTest:
			
			break;
		case ClassificationProtocol::LeaveOneOut:
			break;
		default:
			*/
		}
		
	}

	void ActionRecognition::saveFeature(std::string label, cv::Mat &features, std::string videoName) {
		
		cv::FileStorage storage;
		std::string path;
		// Open modelFile
		path = getFileName(videoName, params);

		storage.open(path, cv::FileStorage::WRITE);
		if (storage.isOpened() == false)
			std::cerr << "Invalid file storage!";

		storage << "ActionRecognitionFeatures" << "{";
		storage << "Label" << label;
		storage << "Features" << features;

		storage.release();
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