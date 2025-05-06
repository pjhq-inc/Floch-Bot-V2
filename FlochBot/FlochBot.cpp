#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::string loadFileData(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << "\n";
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string loadTrainingData() {
    std::string data;
    for (int i = 1; i <= 5; ++i) {
        data += loadFileData("data" + std::to_string(i) + ".txt") + "\n";
    }
    return data;
}

std::string buildPrompt(const std::string& input, const std::string& trainingData) {
    return "You are Floch, obsessed with something called 'Sulhpur evolution' and act like you're good at a game called Skyblock when you're really not. Respond based on the following knowledge:\n"
        + trainingData + "\nUser: " + input + "\nFloch:";
}

std::string runPyTorchModel(torch::jit::script::Module& module, const std::string& prompt) {
    std::vector<int64_t> input_ids = { 101, 102 };

    torch::Tensor input_tensor = torch::tensor(input_ids).unsqueeze(0);  

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    at::Tensor output = module.forward(inputs).toTensor();

    std::string response = "Generated output will be here"; 

    return response;
}

int main() {
    std::string trainingData = loadTrainingData();
    if (trainingData.empty()) {
        std::cerr << "Failed to load data.\n";
        return 1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("gpt2_traced.pt");  
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return 1;
    }

    std::string input;
    while (true && (input == null)) {
        std::cout << "\nYou: ";
        std::getline(std::cin, input);
        if (input == "exit") break;

        std::string prompt = buildPrompt(input, trainingData);
        std::string output = runPyTorchModel(module, prompt);

        std::cout << "Floch: " << output << "\n";
    }

    return 0;
}
