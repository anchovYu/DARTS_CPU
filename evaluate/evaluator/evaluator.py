import argparse
import onnx

import caffe2.python.onnx.backend as backend
# import onnx_caffe2.backend as backend
import numpy as np

parser = argparse.ArgumentParser("evaluator")
parser.add_argument('--onnx_import_path', type=str, default='../model/darts.proto', help='locatin of the imported model onnx file')
args = parser.parse_args()

def main():
  # Load the ONNX model
  model = onnx.load(args.onnx_import_path)

  # Check that the IR is well formed
  onnx.checker.check_model(model)

  # Print a human readable representation of the graph
  print(onnx.helper.printable_graph(model.graph))

  # import to caffe2
  rep = backend.prepare(model, device="CPU")
  outputs = rep.run(np.random.randn(96, 3, 32, 32).astype(np.float32))
  print(outputs)

if __name__ == '__main__':
  main() 
