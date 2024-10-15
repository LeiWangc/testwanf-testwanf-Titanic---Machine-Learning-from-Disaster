#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>
#include <cstdlib>  // 用于随机数生成
#include <ctime>    // 用于设置随机数种子
#include <iostream>
#include <thread>
#include <fstream> // 添加这一行
#include <datareader.h> // 确保包含这个头文件
#include <map>
#include <cmath>   // std::sqrt
#include <numeric> // std::accumulate

std::mutex prepared_flag_mutex;

// 模拟 remove_spaces 函数，移除空格
std::string remove_spaces(const std::string& input) {
    std::string result;
    std::remove_copy_if(input.begin(), input.end(), std::back_inserter(result), ::isspace);
    return result;
}

// // 将字母转为 ASCII 码
// std::vector<int> convert_letters_to_ascii(const std::string& input) {
//     std::vector<int> ascii_values;
//     for (char c : input) {
//         ascii_values.push_back(static_cast<int>(c));
//     }
//     return ascii_values;
// }
// 将字符转换为 ASCII 码的函数，返回 int 向量
std::vector<int> convert_letters_to_ascii(const std::string& input) {
    std::vector<int> result; // 存储结果的整数向量
    for (char c : input) {
        if (std::isdigit(c)) {
            result.push_back(c - '0');  // 如果是数字，转换为 int 并添加
        } else {
            result.push_back(static_cast<int>(c)); // 否则转换为 ASCII 码并添加
        }
    }
    return result; // 返回整数向量
}

// 转换为固定长度向量
// std::vector<int> convert_to_fixed_length_vector(const std::vector<int>& input, size_t length = 10) {
//     std::vector<int> output(length, 0);  // 默认用 0 填充
//     for (size_t i = 0; i < std::min(input.size(), length); ++i) {
//         output[i] = input[i];
//     }
//     return output;
// }
std::vector<int> convert_to_fixed_length_vector(const std::vector<int>& input, size_t length = 10) {
    std::vector<int> output(length, 0);  // 创建一个指定长度的向量，默认用 0 填充

    // 将输入向量的值复制到输出向量的后面
    size_t input_size = input.size();
    if (input_size > length) {
        // 如果输入向量的大小大于输出向量的大小，截取前 length 个元素
        std::copy(input.begin(), input.begin() + length, output.begin());
    } else {
        // 如果输入向量的大小小于或等于输出向量的大小，复制所有元素到输出向量的后面
        std::copy(input.begin(), input.end(), output.begin() + (length - input_size));
    }

    return output;
}

// 模拟 remove_punctuation 函数，移除标点符号
std::string remove_punctuation(const std::string& input) {
    std::string result;
    for (char c : input) {
        if (!std::ispunct(c)) {
            result += c;
        }
    }
    return result;
}
// 计算均值
float compute_mean(const std::vector<float>& data) {
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    // std::cout << "mean:" << sum / data.size()<<std::endl;
    return sum / data.size();
}

// 计算标准差
float compute_std(const std::vector<float>& data, float mean) {
    float sum = 0.0f;
    for (const auto& val : data) {
        sum += (val - mean) * (val - mean);
    }
    // std::cout << "标准差" << std::sqrt(sum / data.size()) <<std::endl;
    return std::sqrt(sum / data.size());
}

// 标准化操作
std::vector<float> normalize(const std::vector<float>& data) {
    float mean = compute_mean(data);
    float std = compute_std(data, mean);
    
    std::vector<float> normalized_data;
    for (const auto& val : data) {
        normalized_data.push_back((val - mean) / std);
    }
    return normalized_data;
}
class my_data{
public:
    std::string file_name;
    my_data(std::string filename) : file_name(filename){
        std::cout << "初始化完成" << std::endl;
    }

    std::vector<std::vector<std::string>> readdata()
    {
    std::vector<std::vector<std::string>> data_frame = read_csv(file_name);
    return data_frame;
    }
private:

    // // 读取 CSV 文件，模拟 pandas DataFrame 的 iloc 功能
    // std::vector<std::vector<std::string>> read_csv(const std::string& filename) {
    //     std::ifstream file(filename);
    //     std::vector<std::vector<std::string>> data;
    //     std::string line;
        
    //     while (std::getline(file, line)) {
    //         std::stringstream ss(line);
    //         std::string item;
    //         std::vector<std::string> row;
    //         while (std::getline(ss, item, ',')) {
    //             row.push_back(item);
    //         }
    //         data.push_back(row);
    //     }
    //     return data;
    // }
    std::vector<std::vector<std::string>> read_csv(const std::string& filename) {
        std::ifstream file(filename);
        std::vector<std::vector<std::string>> data;
        std::string line;
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            std::vector<std::string> row;
            bool inside_quotes = false;  // 用于判断是否在引号内
            std::string current_field;

            while (std::getline(ss, item, ',')) {
                if (inside_quotes) {
                    // 当前处于引号内，将 item 添加到当前字段
                    current_field += "," + item;
                    if (item.back() == '"') {
                        // 遇到结束引号，关闭引号模式
                        inside_quotes = false;
                        current_field.pop_back();  // 移除结束引号
                        row.push_back(current_field);
                        current_field.clear();
                    }
                } else {
                    if (item.front() == '"') {
                        // 遇到开始引号，进入引号模式
                        inside_quotes = true;
                        current_field = item.substr(1);  // 去掉开始引号
                    } else {
                        row.push_back(item);
                    }
                }
            }

            // 处理没有关闭引号的情况
            if (!current_field.empty()) {
                row.push_back(current_field);
            }

            data.push_back(row);
        }

        return data;
}
};

void flow_net(std::string &model_param,std::vector<float> &input,int &flag)//,ncnn::DataReader &model_bin
{
    
    srand(static_cast<unsigned>(time(0)));

    ncnn::Net net;
    if (net.load_param(model_param.c_str()) != 0) {
        std::cerr << "Failed to load param file." << std::endl;
    }
    if (net.load_model("/home/leiwang/workspace/pytorch_learn/kaggle/model.bin") != 0) {
        std::cerr << "Failed to load model file." << std::endl;
    }


    // std::vector<float> input(53);
    while(1)
    {
        if(flag == 0)
        {
            std::cout << "执行推理计算........ " << std::endl;
            // for(int i=0;i<53;i++)
            // {
            //     input[i] = 0.015;//static_cast<float>(rand());  
            // }

            ncnn::Mat input_data = ncnn::Mat(53,input.data());
            ncnn::Extractor flow = net.create_extractor();

            flow.input("input",input_data);
            ncnn::Mat output;
            flow.extract("output",output);
            for (size_t i = 0; i < output.total(); ++i) {
                std::cout << "output:" << ((output[i]>=0.5) ? 1 : 0)<< " " << std::endl;
            }
            flag = 1;
        }else if(flag == 2) break;
    }
}
void data_process(std::vector<float> &input, int& flag)
{
    std::string datapath = "/home/leiwang/workspace/pytorch_learn/kaggle/titanic/test.csv";
    my_data mydata(datapath);
    auto data_frame = mydata.readdata();
    int all_num = data_frame.size();
    int i=1;
    std::cout << "数据长度：" << all_num << std::endl;
    while(1)
    {
        if(flag)
        {
            std::cout << "读取测试集数据........ " << std::endl;
            if(i==all_num) 
            {
                flag = 2;
                std::cout<< "here"<<std::endl;
                break;
            }
            auto item = data_frame[i];
            // for (const auto& dat : item) {
            // std::cout << " " << dat ;
            // }
            // std::cout << std::endl;
            std::string num = item[0];
            std::vector<std::string> feature(item.end()-10,item.end());
            feature[1] = "1";  // 如果需要将 sentence 转换为向量，代码应有所不同
            feature[2] = (feature[2] == "male") ? "0" : "1";
            feature[3] = (feature[3] != " " && !feature[3].empty()) ? feature[3] : "0";

            // 模拟处理其他字段
            std::vector<int> fixed_length_vector = convert_to_fixed_length_vector(convert_letters_to_ascii(remove_spaces(remove_punctuation(feature[feature.size()-4]))),25);
            feature[feature.size()-4] = ""; // 初始化为空字符串
            // 遍历固定长度向量并拼接到 feature 中
            for (const auto& value : fixed_length_vector) {
                feature[feature.size()-4] += std::to_string(value) + " "; // 将每个值转换为字符串并用空格分隔
            }
            // feature[feature.size()-4] = std::to_string(feature[feature.size()-4]);
        
            if (1) {//feature[feature.size() - 2] != " "
                // 转换为固定长度向量并处理
                std::vector<int> fixed_length_vector2 = convert_to_fixed_length_vector(convert_letters_to_ascii(remove_spaces(feature[feature.size()-2])), 20);
                feature[feature.size()-2] = "";
                // 将 vector<int> 转换为字符串
                for (const auto& value : fixed_length_vector2) {
                feature[feature.size()-2] += std::to_string(value) + " "; // 将每个值转换为字符串并用空格分隔
                
            }
            }
            // } else {
            //     feature[feature.size() - 2] = "0"; // 处理为空的情况
            // }

            // 如果需要用 map 进行字典映射
            std::map<char, int> my_dict = {{'C',1},{'Q',2},{'S',3}};  // 假设已经填充了值
            // auto b2 = my_dict.find(feature[feature.size()-1]) ;
            // std::cout << "here " << feature[feature.size()-1][0] << "here" << std::endl;
            feature[feature.size()-1] = ( feature[feature.size()-1] != " " && my_dict.find(feature[feature.size()-1][0]) != my_dict.end()) ? std::to_string(my_dict[feature[feature.size()-1][0]]) : "0";
            
            // 创建并展平特征数据
            std::vector<float> flattened_data;

            for (const auto& elem : feature) {
                std::istringstream iss(elem);
                std::string number;
                
                // 通过空格分割字符串并转换为 float
                while (iss >> number) {
                    flattened_data.push_back(std::stof(number));  // 将每个子字符串转换为 float
                }
            }
            flattened_data[3] = flattened_data[3] * 0.1;
            std::cout << num << std::endl;
            // std::cout << "读取测试集输出：";
            // for (const auto& str : flattened_data) {
            //     std::cout << str << " ";
            // }
            // std::cout << std::endl;
            prepared_flag_mutex.lock();
            input = normalize(flattened_data);
            prepared_flag_mutex.unlock();

            
            flag = 0;
            i++;
        }
    // //
    }
}
int main()
{
    int flowdata_flag = 1;//参数为1时进行数据准备线程，参数为2时进行模型推理线程
    std::string path_p = "/home/leiwang/workspace/pytorch_learn/kaggle/model.param";
    std::string path_b = "/home/leiwang/workspace/pytorch_learn/kaggle/model.bin";
    std::vector<float> input(53);
    std::thread flowing_net(flow_net,std::ref(path_p),std::ref(input),std::ref(flowdata_flag));
    std::thread processing_data(data_process,std::ref(input),std::ref(flowdata_flag));
    processing_data.join();
    flowing_net.join();
    std::cout<< "推理完毕！"<<std::endl;
    return 0;

}