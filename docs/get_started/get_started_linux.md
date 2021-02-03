# Get Started with the OpenVINO™ Toolkit on Linux* {#openvino_docs_get_started_get_started_linux}

This Get Started guide helps you take your first steps with the OpenVINO™ toolkit that you installed on a computer running a Linux* operating system. Before using this guide:

- You must have completed the steps in the guide titled, ["Install Intel® Distribution of OpenVINO™ toolkit for Linux*"](../install_guides/installing-openvino-linux.md), including the:
	* Steps for Intel® Processor Graphics (GPU) if you want to use an installed GPU.
	* Steps for the Intel® Neural Compute Stick 2 if you want to use an installed Intel® Neural Compute Stick 2.
	* Instructions to install the Model Downloader.
- You must have Internet access. If your Internet access is through a proxy server, make sure the proxy information is configured.

This guide provides three demo scripts, one code sample, and three demo applications to help you learn the the OpenVINO workflow described in the [OpenVINO™ Toolkit Overview](https://docs.openvinotoolkit.org/latest/index.html). The variety of samples and demos in this guide help you examine and understand the samples and demos more than using a single large and complex demo or sample.

Upon completion of this guide, you will be able to use sample code and a demo application to:
1. Compile samples from the source files that were installed with the OpenVINO toolkit.
2. Download models and media files.
3. Use the Model Optimizer to convert a trained model to the Intermediate Representation (IR).
4. Locate and identify the IR `.xml` and a `.bin` files.
5. Build an application.
6. Use the Inference Engine to run inference on an IR.
7. Output and view inference results.

This samples and demos have similar fuctions. They both send a formatted Intermediate Representation files through an API to the Inference Engine where they run a neural network model. The applications you use:
- Read audio or video input
- Format the data to suit the model and the Inference Engine
- Determine what to do with the output

The variety of models in the samples and demos have different input requirements and provide different types of output. For example, both the SSD and Yolo models return labels and bounding boxes, but the output between them is different and requires handling specific to the model.

## <a name="openvino-installation"></a>OpenVINO™ toolkit Directory Structure

By default, the Intel® Distribution of OpenVINO™ is installed in:
* For root or administrator: `/opt/intel/openvino_<version>/`
* For non-root and non-administrators: `/home/<USER>/intel/openvino_<version>/`

A symbolic link to the latest installation is `/home/<user>/intel/openvino_2021/`

If you installed the Intel® Distribution of OpenVINO™ toolkit to a directory other than the default, replace `/opt/intel` or `/home/<USER>/` with the directory in which you installed the software.

The primary tools for deploying your models and applications are in `/opt/intel/openvino_2021/deployment_tools`.
<details>
    <summary><strong>Click to see the Intel® Distribution of OpenVINO™ toolkit directory structure</strong></summary>

| Directory&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description                                                                           |  
|:----------------------------------------|:--------------------------------------------------------------------------------------|
| `demo/`                                 | Demo scripts that demonstrate pipelines for inference scenarios. The demo scripts automatically perform steps and print detailed output to the console. For more information, see the [Use OpenVINO: Demo Scripts](#use-openvino-demo-scripts) section.|
| `inference_engine/`                     | Inference Engine directory. Contains Inference Engine API binaries and source files, samples and extensions source files, and resources like hardware drivers.|
| `~intel_models/` | Symbolic link to the `intel_models` subfolder of the `open_model-zoo` folder |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`include/`      | Inference Engine header files. For API documentation, see the [Inference Engine API Reference](./annotated.html). |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`lib/`          | Inference Engine binaries.|
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`samples/`      | Inference Engine samples. Contains source code for C++ and Python* samples and build scripts. See the [Inference Engine Samples Overview](./docs/IE_DG/Samples_Overview.md). |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`src/`          | Source files for CPU extensions.|
| `model_optimizer/`                      | Model Optimizer directory. Contains configuration scripts, scripts to run the Model Optimizer and other files. See the [Model Optimizer Developer Guide](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
| `open_model_zoo/`                       | Open Model Zoo directory. Includes the Model Downloader tool to download [pre-trained OpenVINO](./docs/Pre_Trained_Models.md) and public models, OpenVINO models documentation, demo applications and the Accuracy Checker tool to evaluate model accuracy.|
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`demos/`        | Demo applications for inference scenarios. Also includes documentation and build scripts.| 
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`intel_models/` | Pre-trained OpenVINO models and associated documentation. See the [Overview of OpenVINO™ Toolkit Pre-Trained Models](./docs/Pre_Trained_Models.md).|
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`tools/`        | Model Downloader and Accuracy Checker tools. |
| `tools/`                                | Contains a symbolic link to the Model Downloader folder and auxiliary tools to work with your models: Calibration tool, Benchmark and Collect Statistics tools.|

</details>

## <a name="basic-guidelines-sample-application"></a>Demo and Application Guidelines 

* Before using the OpenVINO™ toolkit, always set up the environment: 
	```sh
	source /opt/intel/openvino_2021/bin/setupvars.sh
	``` 

* When typing commands, you must provide directory paths to binary files, media, and models. To make it easy for you to type these paths, you can move the files to a convenient location.
	
* Include these locations in the directory path:
	- Code sample binary files `~/inference_engine_cpp_samples_build/intel64/Release`
	- Demo and application binary files `~/inference_engine_demos_build/intel64/Release`
	- Media: Video or image. See <a href="#download-media">Download Media</a>.
	- Model: Neural Network topology converted to the IR format (`.bin` and `.xml` files). See <a href="#download-models">Download Models</a>.

## <a name="how-to-use-demos"></a>How to Use the Demo Scripts

Run the demo scripts before using the sample code and applications. The demo scripts create files that you need for the applications.

Each demo follows the same basic steps:
	- Download one or more models.
	- Run the Model Optimizer to convert the model to the IR format.
	- Download videos, audio, or images
	- Run the demo

This guide has instructions for three demo scripts:
- Image Classification - Identifies the type of vehicle displayed in a photograph
- Inference Pipeline - Identifies the license plate characters on a vehicle in a photogaph
- Benchmark - Estimates deep learning inference performance

These scripts show you what it looks like to run inference pipelines with different scenarios. You will see how to: 
* Compile samples from the source files that are delivered as part of the OpenVINO toolkit.
* Download trained models.
* Run pipelines and see the output on the console.

The demo scripts are in `/opt/intel/openvino_<version>/deployment_tools/demo`

You can run the scripts on a [CPU, GPU, Intel Myriad VPU, or HDDL device](https://software.intel.com/en-us/openvino-toolkit/hardware). The default device is CPU. Use the `-d` parameter to use one of the other devices. 

The command to run the scripts looks like this:

```sh
./<script_name> -d [CPU, GPU, MYRIAD, HDDL]
```

Before beginning, view `car.bmp` and `car_1.bmp` in the `/opt/intel/openvino_<version>/deployment_tools/demo` directory. The scipts use these photographs and it will be helpful to see what the original images looks like.

### Image Classification Script
The `demo_squeezenet_download_convert_run` script runs the Image Classification demo. This demo illustrates the image classification pipeline.

This script: 
1. Downloads a SqueezeNet model. 
2. Runs the Model Optimizer to convert the model to the IR.
3. Builds the Image Classification Sample Async application.
4. Runs the compiled sample with the `car.png` image that is in the `demo` directory.

<details>
    <summary><strong>Click for the instructions to run the Image Classification script</strong></summary>

The command line for this script uses a CPU. Since CPU is the default device, you don't need to use the `-d` parameter to specify a device:
	```sh
	./demo_squeezenet_download_convert_run.sh
	```
When the script completes, you see the label and confidence for the top-10 categories. This shows you what type of car is likely shown in `car.png`. In this case, it is likely that the `car.png` is likely a sports car, and it took approximately 2.7 microseconds to determine this information:

```sh

Top 10 results:

Image /home/user/dldt/inference-engine/samples/sample_data/car.png

classid probability label
------- ----------- -----
817     0.8363345   sports car, sport car
511     0.0946488   convertible
479     0.0419131   car wheel
751     0.0091071   racer, race car, racing car
436     0.0068161   beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
656     0.0037564   minivan
586     0.0025741   half track
717     0.0016069   pickup, pickup truck
864     0.0012027   tow truck, tow car, wrecker
581     0.0005882   grille, radiator grille


total inference time: 2.6642941
Average running time of one iteration: 2.6642941 ms

Throughput: 375.3339402 FPS

[ INFO ] Execution successful
```

</details>

### Inference Pipeline script
The `demo_security_barrier_camera` script runs the Inference Pipeline demo. This demo uses vehicle recognition with three models that build on each other to provide information about a specific attribute. 

The `demo_security_barrier_camera` script:
1. Downloads three pre-trained model IRs.
2. Builds the Security Barrier Camera demo application.
3. Runs the application with the downloaded models and the `car_1.bmp` image from the `demo` directory to show an inference pipeline. 

The Security Barrier Camera demo application:

1. Identifies an object as a vehicle. 
2. Uses the vehicle identification as input to the second model to identify the license plate.
3. Uses the the license plate as input to the third model to recognize specific characters in the license plate.

<details>
    <summary><strong>Click for the instructions to run the Pipeline script</strong></summary>
    
The command line for this script uses Intel® Processor Graphics (GPU). If you aren't using a GPU, change the `-d` parameter in the command line. See <a href="#how-to-use-demos">How to Use the Demos for your other options.</a>
```sh
./demo_security_barrier_camera.sh -d GPU
```
When the application completes, you see the modified `car_1.bmp`. The image displays text and a frame with detections rendered as bounding boxes:

![](../img/inference_pipeline_script_lnx.png)

</details>

### Benchmark Demo
The `demo_benchmark_app` script runs the Benchmark demo. This script shows you how to use the Benchmark application to estimate deep learning inference performance. 

The `demo_benchmark_app` script: 
1. Downloads a SqueezeNet model.
2. Runs the Model Optimizer to convert the model to the IR.
3. Builds the Inference Engine Benchmark tool.
4. Runs the tool with the `car.png` image located in the `demo` directory.

<details>
    <summary><strong>Click for the instructions to run the Benchmark demo</strong></summary>

The command line for this script uses the Intel® Vision Accelerator Design with an Intel® Movidius™ VPU. If you aren't using this HDDL device, change the `-d` parameter in the command line. </a>

```sh
./demo_squeezenet_download_convert_run.sh -d HDDL
```
When the verification script completes, the screen displays the performance counters, resulting latency, and throughput values.
</details>

## <a name="using-sample-application"></a>How to Use the Code Samples and Demo Applications

You will perform the following steps in this section: 

1. <a href="#build-samples">Build the code samples and demo applications.</a>
1. <a href="#download-models">Use the Model Downloader to download suitable models.</a>
2. <a href="#convert-models-to-intermediate-representation">Convert the models with the Model Optimizer.</a> 
3. <a href="#download-media">Download media files to run inference on.</a>
4. <a href="#run-image-classification">Run inference on the Image Classification Code Sample and see the results</a>. 
5. <a href="#run-security-barrier">Run inference on the Security Barrier Camera Demo application and see the results</a>.

Each code sample and demo application follow the same behavior and use the same components. 

[Code Samples](../IE_DG/Samples_Overview.html) are small console applications that show how to use specific OpenVINO capabilities within an application and specific tasks like loading a model, running inference, querying specific device capabilities, and more.

[Demo Applications](@ref omz_demos_README) are console applications that provide templates to help you implement  deep learning scenarios. These templates can involve complex processing pipelines that gather analysis from several models that run inference simultaneously, such as detecting a person in a video stream along with that person's attributes, like age, gender, and emotions.
 
Inputs you must specify:
- <b>A compiled OpenVINO™ code sample or demo application</b> that runs inference on a model that has been converted to the IR format. 
- <b>One or more models</b> in the Intermediate Representation format. Each model is trained for a specific task. Examples include pedestrian detection, face detection, vehicle detection, license plate recognition, head pose, and others. Different models are used for different applications. You can chain models together to provide multiple identifications, such as a vehicle + the make/model of the vehicle + the license plante of the vehicle.
- <b>One or more media files</b>. The media is usually a video file, but it can be a still photo.
- <b>One or more target devices</b> on which you run inference. The target device can be the CPU, GPU, or VPU accelerator.

### <a name="build-samples"></a> Step 1: Build the Code Samples and Demo Applications 

Use Image Classification code sample and the Security Barrier Camera demo application that were compiled when you ran the Image Classification and Inference Pipeline demo scripts. The binary files are in the `~/inference_engine_cpp_samples_build/intel64/Release` and `~/inference_engine_demos_build/intel64/Release` directories, respectively.

To run other sample code or demo applications, build them from the source files that were delivered as part of the OpenVINO toolkit. 

To learn how to build code samples see the [Inference Engine Code Samples Overview](../IE_DG/Samples_Overview.md) To learn how to build demo applications see the [Demo Applications Overview](@ref omz_demos_README).

### <a name="download-models"></a> Step 2: Download the Models

You must have a model that is specific for your inference task. Example model types are:
- Classification (AlexNet, GoogleNet, SqueezeNet, others) - Detects one type of element in a frame.
- Object Detection (SSD, YOLO) - Draws bounding boxes around multiple types of objects.
- Custom (Often based on SSD)

Your options to find a model suitable for the OpenVINO™ toolkit are:
- Use [Model Downloader tool](@ref omz_tools_downloader_README) Intel's or publicly available pre-trained models from the [Open Model Zoo](https://github.com/opencv/open_model_zoo). 
- Download models from GitHub*, Caffe* Zoo, TensorFlow* Zoo, or other sources.
- Train your own model.

These instructions direct you to use the Model Downloader to download the following:

|Model Name                                     | Code Sample or Demo App                             |
|-----------------------------------------------|-----------------------------------------------------|
|`squeezenet1.1`                                | Image Classification Sample                         |
|`vehicle-license-plate-detection-barrier-0106` | Security Barrier Camera Demo application            |
|`vehicle-attributes-recognition-barrier-0039`  | Security Barrier Camera Demo application            |
|`license-plate-recognition-barrier-0001`       | Security Barrier Camera Demo application            |


<details><summary><b>Click here for instuctions to download the models for the squeezenet1.1 sample</b></summary>

Use the Model Downloader to get the following models and put them in the `~/models` folder:

	```sh
	sudo python3 ./downloader.py --name squeezenet1.1 --output_dir ~/models
	```

Your screen looks similar to this after the download:

```
###############|| Downloading models ||###############

========= Downloading /home/username/models/public/squeezenet1.1/squeezenet1.1.prototxt

========= Downloading /home/username/models/public/squeezenet1.1/squeezenet1.1.caffemodel
... 100%, 4834 KB, 3157 KB/s, 1 seconds passed

###############|| Post processing ||###############

========= Replacing text in /home/username/models/public/squeezenet1.1/squeezenet1.1.prototxt =========
```
</details>

<details>
    <summary><strong><b>Click for the instructions to download the models for the Security Barrier Camera demo</b></strong></summary>

To download all three pre-trained models in FP16 precision to the `~/models` folder:   

```sh
./downloader.py --name vehicle-license-plate-detection-barrier-0106,vehicle-attributes-recognition-barrier-0039,license-plate-recognition-barrier-0001 --output_dir ~/models --precisions FP16
```   
Your screen looks similar to this after the download:
```
################|| Downloading models ||################

========== Downloading /home/username/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml
... 100%, 204 KB, 183949 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin
... 100%, 1256 KB, 3948 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml
... 100%, 32 KB, 133398 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.bin
... 100%, 1222 KB, 3167 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.xml
... 100%, 47 KB, 85357 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.bin
... 100%, 2378 KB, 5333 KB/s, 0 seconds passed

################|| Post-processing ||################
```
</details>

### <a name="convert-models-to-intermediate-representation"></a> Step 3: Convert the Models to the Intermediate Representation

In this step, you use the Model Optimizer to convert your trained models to the Intermediate Representation (IR) format. The IR is a pair of `.xml` and `.bin` files:
- `model_name.xml`
- `model_name.bin`

You can't do inference on your model until you have these two files.

This guide uses the public SqueezeNet 1.1 Caffe\* model to run the Image Classification Sample. See the example <a href="#download-models">Download Models</a> for instructions to download this model.

<b>About the models:</b>
- The `squeezenet1.1` model is in the Caffe* format. You will use the Model Optimizer to convert the model to the IR format. 
- The `vehicle-license-plate-detection-barrier-0106`, `vehicle-attributes-recognition-barrier-0039`,  and`license-plate-recognition-barrier-0001` models are already in the Intermediate Representation format. You don't need to use the Model Optimizer to convert these models.

1. Create an `<ir_dir>` directory to contain the model's Intermediate Representation (IR). 

2. The Inference Engine can perform inference on different precision formats, such as `FP32`, `FP16`, `INT8`. To prepare an IR with specific precision, run the Model Optimizer with the appropriate `--data_type` option.

   ```sh
   cd /opt/intel/openvino_2021/deployment_tools/model_optimizer
   ```
   ```sh  
   python3 ./mo.py --input_model <model_dir>/<model_file> --data_type <model_precision> --output_dir <ir_dir>
   ```
   The IR files are created in the `<ir_dir>` directory.

<details>
    <summary><strong>Click for the instructions to convert the SqueezeNet Caffe* model</strong></summary>

The following command converts the public SqueezeNet 1.1 Caffe\* model to the FP16 IR and saves the IR to the `~/models/public/squeezenet1.1/ir` output directory:

	```sh
	cd /opt/intel/openvino_2021/deployment_tools/model_optimizer
	```
	```sh  
	python3 ./mo.py --input_model ~/models/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP16 --output_dir ~/models/public/squeezenet1.1/ir
	```

   The IR files are created in the `~/models/public/squeezenet1.1/ir` directory.

Copy the `squeezenet1.1.labels` file from the `/opt/intel/openvino_2021/deployment_tools/demo/` to `<ir_dir>`. `squeezenet1.1.labels` contains the classes that ImageNet uses. Therefore, the inference results show text instead of classification numbers:
   ```sh   
   cp /opt/intel/openvino_2021/deployment_tools/demo/squeezenet1.1.labels <ir_dir>
   ```
</details>

### <a name="download-media"></a> Step 4: Download a Video or a Still Photo as Media

Many sources are available from which you can download video media to use the code samples and demo applications. Possibilities include: 
- https://videos.pexels.com
- https://images.google.com

For the exercises in this guide, the Intel® Distribution of OpenVINO™ toolkit includes two sample images that you can use for running code samples and demo applications:
* `/opt/intel/openvino_2021/deployment_tools/demo/car.png`
* `/opt/intel/openvino_2021/deployment_tools/demo/car_1.bmp`

### <a name="run-image-classification"></a>Step 5: Run the Image Classification Code Sample

> <b>NOTE</b>: The Image Classification code sample is compiled when you run the Image Classification demo script. If you want to compile it manually, see the [Inference Engine Code Samples Overview](../IE_DG/Samples_Overview.html#build_samples_linux) section. 

To run the <b>Image Classification</b> code sample with an input image on the IR: 

1. Set up the OpenVINO environment variables:
   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ``` 
2. Go to the code samples build directory:
   ```sh
   cd ~/inference_engine_samples_build/intel64/Release
   ```
3. Run the code sample executable. In the command line specify the input media file, the IR of your model, and a target device on which you want to perform inference:
   ```sh
   classification_sample_async -i <path_to_media> -m <path_to_model> -d <target_device>
   ```
<details>
    <summary><strong>Click for the instructions to run the Image Classification code sample on different devices</strong></summary>

The following commands run the Image Classification Code Sample using the `car.png` file from the `/opt/intel/openvino_2021/deployment_tools/demo/` directory as an input image, the IR of your model from `~/models/public/squeezenet1.1/ir` and on different hardware devices:

<b>CPU</b>
   ```sh
   ./classification_sample_async -i /opt/intel/openvino_2021/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d CPU
   ```

<b>GPU</b>
	```sh
	./classification_sample_async -i /opt/intel/openvino_2021/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d GPU
	```
   
<b>MYRIAD (Intel® Neural Compute Stick 2)</b> 
	```sh   
	./classification_sample_async -i /opt/intel/openvino_2021/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d MYRIAD
	```
   
<b>HDDL (Intel® Vision Accelerator Design with Intel® Movidius™ VPUs device with the HDDL plugin)</b>
	```sh   
	./classification_sample_async -i /opt/intel/openvino_2021/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d HDDL
	```

When the Sample Application completes, you see the label and confidence for the top-10 categories. Below is a sample output with inference results on CPU:    
```sh
Top 10 results:

Image /home/user/dldt/inference-engine/samples/sample_data/car.png

classid probability label
------- ----------- -----
817     0.8363345   sports car, sport car
511     0.0946488   convertible
479     0.0419131   car wheel
751     0.0091071   racer, race car, racing car
436     0.0068161   beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
656     0.0037564   minivan
586     0.0025741   half track
717     0.0016069   pickup, pickup truck
864     0.0012027   tow truck, tow car, wrecker
581     0.0005882   grille, radiator grille

total inference time: 2.6642941
Average running time of one iteration: 2.6642941 ms

Throughput: 375.3339402 FPS

[ INFO ] Execution successful
```

</details>

### <a name="run-security-barrier"></a>Step 6: Run the Security Barrier Camera Demo Application

The Security Barrier Camera Demo Application was compiled when you ran the Inference Pipeline demo. If you want to build it manually, see the [Demo Applications Overview](@ref omz_demos_README).

To run the <b>Security Barrier Camera Demo Application</b> with an input image and the prepared IRs:

1. Set up the OpenVINO environment variables:
   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ``` 
2. Go to the demo application build directory:
   ```sh
   cd ~/inference_engine_demos_build/intel64/Release
   ```
3. Run the demo executable, specifying the input media file, list of model IRs, and a target device on which to perform inference:
   ```sh
   ./security_barrier_camera_demo -i <path_to_media> -m <path_to_vehicle-license-plate-detection_model_xml> -m_va <path_to_vehicle_attributes_model_xml> -m_lpr <path_to_license_plate_recognition_model_xml> -d <target_device>
   ```

<details>
    <summary><strong>Click for the instructions to run the Security Barrier Camera demo application on different devices</strong></summary>

<b>CPU</b>

```sh
./security_barrier_camera_demo -i /opt/intel/openvino_2021/deployment_tools/demo/car_1.bmp -m /home/username/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml -m_va /home/username/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml -m_lpr /home/username/models/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.xml -d CPU
```

<b>GPU (Intel® Processor Graphics)</b>
	```sh
	./security_barrier_camera_demo -i /opt/intel/openvino_2021/deployment_tools/demo/car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU
	```

<b>MYRIAD (Intel® Neural Compute Stick 2)</b> 
	```sh   
	./classification_sample_async -i <DLDT_DIR>/inference-engine/samples/sample_data/car.png -m <ir_dir>/squeezenet1.1.xml -d MYRIAD
	```

<b>HDDL (Intel® Vision Accelerator Design with Intel® Movidius™ VPUs device with the HDDL plugin)</b>
	```sh   
	./classification_sample_async -i <DLDT_DIR>/inference-engine/samples/sample_data/car.png -m <ir_dir>/squeezenet1.1.xml -d HDDL
	```
</details>


## <a name="syntax-examples"></a>Code Sample and Demo Application Syntax Examples

Template to call sample code or a demo application:

```sh
<path_to_app> -i <path_to_media> -m <path_to_model> -d <target_device>
```

With sample information specified, the command might look like this:

```sh
./object_detection_demo_ssd_async -i ~/Videos/catshow.mp4 \
-m ~/ir/fp32/mobilenet-ssd.xml -d CPU
```

## <a name="advanced-samples"></a>Advanced Demo Use 

Some demo applications let you use multiple models for different purposes. In these cases, the output of the first model is usually used as the input for later models.

For example, an SSD will detect a variety of objects in a frame, then age, gender, head pose, emotion recognition and similar models target the objects classified by the SSD to perform their functions.

In these cases, the use pattern in the last part of the template above is usually:

`-m_<acronym> … -d_<acronym> …`

For head pose:

`-m_hp <headpose model> -d_hp <headpose hardware target>`

<b>Example of an Entire Command (object_detection + head pose)<b>:

```sh
./object_detection_demo_ssd_async -i ~/Videos/catshow.mp4 \
-m ~/ir/fp32/mobilenet-ssd.xml -d CPU -m_hp headpose.xml \
-d_hp CPU
``` 

<b>Example of an Entire Command (object_detection + head pose + age-gender)</b>:

```sh
./object_detection_demo_ssd_async -i ~/Videos/catshow.mp4 \
-m ~/r/fp32/mobilenet-ssd.xml -d CPU -m_hp headpose.xml \
-d_hp CPU -m_ag age-gender.xml -d_ag CPU
```

You can see all the sample application’s parameters by adding the `-h` or `--help` option at the command line.


## Additional Resources

Use these resources to learn more about the OpenVINO™ toolkit:

* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [OpenVINO™ Toolkit Overview](../index.md)
* [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
* [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md)
* [Overview of OpenVINO™ Toolkit Pre-Trained Models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)
* [OpenVINO™ Hello World Face Detection Exercise](https://github.com/intel-iot-devkit/inference-tutorials-generic)
