
#include "classification.hpp"

namespace ccr
{
	Classification::Classification(cv::FileStorage &storage) {
		nsamples = 0;
		params = storage;
	}


	Classification::~Classification() {}

	void Classification::save(std::string filename) {
		cv::FileStorage storage;

		storage.open(filename, cv::FileStorage::WRITE);
		this->save(storage);
		storage.release();
	}


	void Classification::load(std::string filename) {
		cv::FileStorage storage;

		storage.open(filename, cv::FileStorage::READ);
		this->load(storage.root(), storage);
		storage.release();
	}


	void Classification::addSamples(const cv::Mat &X, std::string id) {
		int i;

		dataX.push_back(X);
		nsamples += X.rows;

		for (i = 0; i < X.rows; i++)
			dataY.push_back(id);

		labels.insert(id);
	}


	// add sample features to the classifier
	void Classification::addSamples(const cv::Mat &X, const std::vector<std::string> &Y) {
		int i;
		if (X.rows != (int)Y.size())
			std::cerr << "Inconsistent matrix sizes (number of rows in X and Y must be the same)";

		dataX.push_back(X);

		for (i = 0; i < (int)Y.size(); i++) {
			dataY.push_back(Y[i]);
			labels.insert(Y[i]);
		}

		nsamples += X.rows;
	}

	void Classification::addExtraSamples(const cv::Mat &X, const std::vector<std::string> &Y){
		int i;
		if (X.rows != (int)Y.size())
			std::cerr << "Inconsistent matrix sizes (number of rows in X and Y must be the same)";
		dataX.push_back(X);
		for (i = 0; i < (int)Y.size(); i++) {
			extraDataY.push_back(Y[i]);
			extraLabels.insert(Y[i]);
		}
	}

	void Classification::addExtraSamples(const cv::Mat &X) {
		int i;

		dataX.push_back(X);
		nsamples += X.rows;

		for (i = 0; i < X.rows; i++)
			dataY.push_back(EXTRA_CLASS);

		labels.insert(EXTRA_CLASS);
	}

	void Classification::addExtraSamples(const cv::Mat &X, std::string id){
		int i;
		extraDataX.push_back(X);

		for (i = 0; i < X.rows; i++)
			extraDataY.push_back(id);
		extraLabels.insert(id);
	}

	std::vector<std::string> Classification::retrieveClassIDs() {
		std::set<std::string>::iterator it;
		std::vector<std::string> ids;

		for (it = labels.begin(); it != labels.end(); it++) {
			ids.push_back(*it);
		}

		return ids;
	}

	std::vector<int> Classification::retrieveClassIntIDs() {
		std::vector<std::string> ids;
		std::string id;
		std::vector<int> int_ids;
		size_t i, j;

		ids = this->retrieveClassIDs();

		for (i = 0; i < dataY.size(); i++) {
			id = dataY[i];
			for (j = 0; j < ids.size(); j++) {
				if (ids[j] == id)
					break;
			}
			int_ids.push_back(static_cast<int>(j));
		}

		return int_ids;
	}


	int Classification::retrieveResponseClassIDPosition(std::string id) {
		std::vector<std::string> ids;
		size_t i;

		ids = this->retrieveClassIDs();

		for (i = 0; i < ids.size(); i++) {
			if (id == ids[i])
				return (int)i;
		}

		std::cerr << "Invalid id " << id.c_str() << " for this classification meodel";
		return -1;
	}



	void Classification::save_(cv::FileStorage &storage) {
		std::set<std::string>::iterator it;

		storage << "ClassificationMethod" << "{";
		storage << "labels" << "[:";
		for (it = labels.begin(); it != labels.end(); it++)
			storage << *it;
		storage << "]";
		storage << "}";
	}

	//Must be tested
	void Classification::load_(const cv::FileNode &node) {
		cv::FileNode n, n1;
		std::vector<std::string> tmp;
		size_t i;

		n = node["ClassificationMethod"];
		n1 = n["labels"];

		cv::FileNodeIterator it = n1.begin(), it_end = n1.end();

		// iterate through a sequence using FileNodeIterator
		for (; it != it_end; ++it)
			labels.insert((std::string)(*it));
	}

	void Classification::reset() {

		nsamples = 0;
		dataX.release();
		dataY.clear();
		labels.clear();
	}


}
