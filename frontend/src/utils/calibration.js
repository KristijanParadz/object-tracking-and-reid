export function checkIfCameraHasIntrinsics(camIndex) {
  const calibrationList =
    JSON.parse(localStorage.getItem("calibrationData")) || [];

  if (!Array.isArray(calibrationList)) {
    return false;
  }

  const cameraData = calibrationList.find((item) => item.index === camIndex);

  if (
    !cameraData ||
    !Array.isArray(cameraData.K) ||
    !Array.isArray(cameraData.distCoef)
  ) {
    return false;
  }

  // Check if K is a 3x3 matrix
  const isKValid =
    cameraData.K.length === 3 &&
    cameraData.K.every(
      (row) =>
        Array.isArray(row) &&
        row.length === 3 &&
        row.every((num) => typeof num === "number")
    );

  // Check if distCoef is an array of 5 numbers
  const isDistCoefValid =
    cameraData.distCoef.length === 5 &&
    cameraData.distCoef.every((num) => typeof num === "number");

  return isKValid && isDistCoefValid;
}

export function checkIfCameraHasExtrinsics(camIndex) {
  const calibrationList =
    JSON.parse(localStorage.getItem("calibrationData")) || [];

  if (!Array.isArray(calibrationList)) {
    return false;
  }

  const cameraData = calibrationList.find((item) => item.index === camIndex);

  if (
    !cameraData ||
    !Array.isArray(cameraData.R) ||
    !Array.isArray(cameraData.t)
  ) {
    return false;
  }

  // Check if R is a 3x3 matrix
  const isRValid =
    cameraData.R.length === 3 &&
    cameraData.R.every(
      (row) =>
        Array.isArray(row) &&
        row.length === 3 &&
        row.every((num) => typeof num === "number")
    );

  // Check if t is a 3x1 matrix
  const isTValid =
    cameraData.t.length === 3 &&
    cameraData.t.every(
      (row) =>
        Array.isArray(row) && row.length === 1 && typeof row[0] === "number"
    );

  return isRValid && isTValid;
}
