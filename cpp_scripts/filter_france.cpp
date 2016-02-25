#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>


using namespace std;

void parse_file(string filename) {
    ifstream file(filename);
    string line, token;
    vector<float> res;

    while (getline(file, line)) {
        stringstream ss(line);
        while (getline(ss, token, ' ')) {
            res.push_back(float(atof(token.c_str())));
        }
//        cout << line << endl;
    }

    for (float f : res) {
        cout << f << " ";
    }
}

int main() {
    string filename("files/train_2011_2012.csv");
    ifstream file(filename);
    ofstream out("files/train_small.csv");
    ofstream small("files/train_small.csv");

    auto rand = uniform_real_distribution<double>(0.0, 1.0);
    auto gen = default_random_engine();

    string line, token;
    getline(file, line);
    out << line << endl;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> tokens;
        while (getline(ss, token, ';')) {
            tokens.push_back(token);
        }

        if (tokens[10] == "Entity1 France" && rand(gen) < 0.02) {
            out << tokens[0];
            for (int i = 1; i < tokens.size(); i++) {
                out << ';' << tokens[i];
            }
            out << endl;
        }
    }

    out.close();
    return 0;
}