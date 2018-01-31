#define WIN32_LEAN_AND_MEAN
#include <dlib\dnn.h>
#include <iostream>
#include <dlib\data_io.h>
#include <stdlib.h>     /* srand, rand */
#include <fstream>

using namespace dlib;

std::vector<matrix<int>> generateTrainingData(std::vector<float>& getOutput) {
	std::vector<matrix<int>> vec;
	srand(600000);
	for (int i = 0; i < 15000; i++) {
		int num = std::rand() % 1000 + 1;
		int num2 = std::rand() % 1000 + 1;

		matrix<int, 1, 2> mat;
		mat = num, num2;
		if (i < 10000) {
			if (num > 500 && num2 > 300) {
				getOutput.push_back(1.0f);
			}
			else {
				getOutput.push_back(-1.0f);
			}
		}
		vec.push_back(mat);
	}
	return vec;
}

int main() {
	std::vector<matrix<int>> all_data;
	std::vector<matrix<int>> training_data;
	std::vector<float> training_labels;
	std::vector<matrix<int>> testing_data;

	using net_type = loss_binary_hinge < fc<1, fc<3,
		fc < 5,
		input<
		matrix<int
		>>>>>>;
	net_type net;
	dnn_trainer<net_type> trainer(net);
	trainer.set_learning_rate(0.001);
	trainer.set_min_learning_rate(0.00001);
	trainer.be_verbose();
	trainer.set_max_num_epochs(400);
	all_data = generateTrainingData(training_labels);

	for (int i = 0; i < 10000; i++) {
		training_data.push_back(all_data[i]);
	}

	for (int i = 10000; i < 15000; i++) {
		testing_data.push_back(all_data[i]);
	}
	trainer.train(training_data, training_labels);

	auto predicted = net(testing_data);

	int wrongCount = 0;
	for (int i = 0; i < testing_data.size(); i++) {
		if (predicted[i] < 0)
		{
			if (testing_data[i](0) > 500 && testing_data[i](1) > 300)
			{
				wrongCount++;
			}
		}
		else {
			if (testing_data[i](0) <= 500 || testing_data[i](1) <= 300) {
				wrongCount++;
			}
		}
	}

	std::cout << "Percent of false positives " << (double)wrongCount / testing_data.size();
	system("pause");
}