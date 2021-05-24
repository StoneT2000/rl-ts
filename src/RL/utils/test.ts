import { Tensor, TensorLike, gather,  } from "@tensorflow/tfjs-core";

function booleanMaskAsync_(
  tensor: Tensor|TensorLike, mask: Tensor|TensorLike,
  axis?: number): Promise<Tensor> {
const $tensor = convertToTensor(tensor, 'tensor', 'boolMask');
const $mask = convertToTensor(mask, 'mask', 'boolMask', 'bool');

const axisFrom = axis == null ? 0 : axis;
const maskDim = $mask.rank;
const tensorShape = $tensor.shape;

util.assert(maskDim > 0, () => 'mask cannot be scalar');
util.assertShapesMatch(
    tensorShape.slice(axisFrom, axisFrom + maskDim), $mask.shape,
    `mask's shape must match the first K dimensions of tensor's shape,`);

let leadingSize = 1;
for (let i = axisFrom; i < axisFrom + maskDim; i++) {
  leadingSize *= tensorShape[i];
}
const targetTensorShape =
    tensorShape.slice(0, axisFrom)
        .concat([leadingSize], tensorShape.slice(axisFrom + maskDim));
const reshapedTensor = reshape($tensor, targetTensorShape);
const reshapedMask = reshape($mask, [-1]);
const positivePositions = await whereAsync(reshapedMask);
const indices = squeeze(positivePositions, [1]);

const res = gather(reshapedTensor, indices, axisFrom);

// Ensure no memory leak.
if (tensor !== $tensor) {
  $tensor.dispose();
}
if (mask !== $mask) {
  $mask.dispose();
}
indices.dispose();
reshapedTensor.dispose();
reshapedMask.dispose();
positivePositions.dispose();

return res;
}
