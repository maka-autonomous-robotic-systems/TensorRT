LD_LIBRARY_PATH=./build/

./build/trtexec_parallel --deploy=/usr/src/tensorrt/data/mnist/deploy.prototxt --model=/usr/src/tensorrt/data/mnist/mnist.caffemodel --output=prob --batch=16 --saveEngine=mnist16.trt --device=1
./build/trtexec_parallel --deploy=/usr/src/tensorrt/data/mnist/deploy.prototxt --model=/usr/src/tensorrt/data/mnist/mnist.caffemodel --output=prob --batch=16 --saveEngine=mnist16.trt --device=2
echo "Running multi-threaded load"
time ./build/trtexec_parallel --deploy=/usr/src/tensorrt/data/mnist/deploy.prototxt --model=/usr/src/tensorrt/data/mnist/mnist.caffemodel --output=prob --batch=16 --loadEngine=mnist16.trt --parallel --loadTwo
echo "Running sequential load"
time ./build/trtexec_parallel --deploy=/usr/src/tensorrt/data/mnist/deploy.prototxt --model=/usr/src/tensorrt/data/mnist/mnist.caffemodel --output=prob --batch=16 --loadEngine=mnist16.trt  --loadTwo

prog1="./build/trtexec_parallel --deploy=/usr/src/tensorrt/data/mnist/deploy.prototxt --model=/usr/src/tensorrt/data/mnist/mnist.caffemodel --output=prob --batch=16 --loadEngine=mnist16.trt --device=1"
prog2="./build/trtexec_parallel --deploy=/usr/src/tensorrt/data/mnist/deploy.prototxt --model=/usr/src/tensorrt/data/mnist/mnist.caffemodel --output=prob --batch=16 --loadEngine=mnist16.trt --device=2"

echo "Running multi-process load"
time ${prog1} & ${prog2} && wait