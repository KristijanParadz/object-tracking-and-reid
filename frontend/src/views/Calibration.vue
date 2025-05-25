<script setup>
import { ref, onMounted, watch } from "vue";
import { socket, intrinsicLiveFeedState } from "@/socket";
import axios from "axios";

const cameras = ref([]);
const selectedCamera = ref(null);
const imagesPreview = ref(null);
const isProcessRunning = ref(false);

async function fetchAvailableCameras() {
  const response = await axios.get(
    `${import.meta.env.VITE_API_BASE_URL}/api/available-cameras`
  );
  cameras.value = response.data.map((camIndex) => {
    return {
      index: camIndex,
      name: `Camera ${camIndex}`,
      isCalibrated: checkIfCameraIsCalibrated(camIndex),
    };
  });
}

function checkIfCameraIsCalibrated(camIndex) {
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

watch(
  () => intrinsicLiveFeedState.framesSaved,
  () => {
    if (intrinsicLiveFeedState.framesSaved === 10) getCapturedImagesPreview();
  }
);

async function getCapturedImagesPreview() {
  const response = await axios.get(
    `${import.meta.env.VITE_API_BASE_URL}/api/intrinsic-images-preview`
  );
  imagesPreview.value = response.data;
}

function selectCamera(index) {
  if (isProcessRunning.value) return;
  selectedCamera.value = cameras.value.find((camera) => camera.index === index);
}

function startCalibration() {
  isProcessRunning.value = true;
  socket.emit("start-intrinsic-calibration", {
    camera_index: selectedCamera.value.index,
  });
}

async function calibrateCamera() {
  const response = await axios.get(
    `${import.meta.env.VITE_API_BASE_URL}/api/intrinsic-camera-calibration`
  );
  const data = response.data;

  // Retrieve and parse existing calibration data from localStorage
  let calibrationList =
    JSON.parse(localStorage.getItem("calibrationData")) || [];

  // Ensure it's an array
  if (!Array.isArray(calibrationList)) {
    calibrationList = [];
  }

  // Find index of existing calibration entry
  const existingIndex = calibrationList.findIndex(
    (item) => item.index === selectedCamera.value.index
  );

  if (existingIndex !== -1) {
    // Update existing entry
    calibrationList[existingIndex].K = data.K;
    calibrationList[existingIndex].distCoef = data.distCoef;
  } else {
    // Create new entry
    calibrationList.push({
      index: selectedCamera.value.index,
      K: data.K,
      distCoef: data.distCoef,
    });
  }

  // Sort list by index
  calibrationList.sort((a, b) => a.index - b.index);

  // Store back into localStorage
  localStorage.setItem("calibrationData", JSON.stringify(calibrationList));
  restartProcess();
}

function captureImage() {
  socket.emit("intrinsic-request-frame-save", {
    frame_number: intrinsicLiveFeedState.frameNumber,
  });
}

function restartProcess() {
  fetchAvailableCameras();
  imagesPreview.value = null;
  intrinsicLiveFeedState.image = null;
  intrinsicLiveFeedState.frameNumber = null;
  intrinsicLiveFeedState.framesSaved = null;
  isProcessRunning.value = false;
}

onMounted(() => {
  fetchAvailableCameras();
});
</script>

<template>
  <main>
    <img src="../assets/protostar-logo.png" alt="protostar-logo" />

    <div class="int-ext-container">
      <div class="int-ext-text">Intrinsic</div>
      <div class="int-ext-text">Extrinsic</div>
    </div>

    <div class="available-cameras-container">
      <span class="text-bold available-cameras-text">Available Cameras</span>
      <div v-if="cameras.length > 0" class="camera-list">
        <div
          v-for="camera in cameras"
          :class="`available-camera ${
            camera.index === selectedCamera?.index && 'selected'
          }`"
          @click="() => selectCamera(camera.index)"
        >
          {{ `${camera.name} ${camera.isCalibrated ? "je" : "nije"}` }}
        </div>
      </div>

      <div v-else class="no-cameras-text">
        No cameras are currently available. Please connect your cameras.
      </div>

      <button
        @click="startCalibration"
        :disabled="selectedCamera == null || isProcessRunning"
      >
        Start Calibration
      </button>
    </div>

    <div class="container">
      <div class="camera-container">
        <div class="image-title-container">
          <span class="text-bold">{{
            imagesPreview ? "Preview" : "Live Feed"
          }}</span>
          <span>{{ intrinsicLiveFeedState.framesSaved || 0 }} / 10</span>
        </div>

        <div v-if="imagesPreview" class="images-container">
          <div v-for="image in imagesPreview" class="image-container">
            <img
              :src="`data:image/jpg;base64,${image}`"
              alt="input"
              class="input-image"
            />
          </div>
          <button @click="calibrateCamera">Calibrate</button>
          <button @click="restartProcess">Try again</button>
        </div>

        <div v-else-if="intrinsicLiveFeedState.image">
          <div class="image-container">
            <img
              :src="`data:image/jpg;base64,${intrinsicLiveFeedState.image}`"
              alt="input"
              class="input-image"
            />
          </div>
          <button @click="captureImage">Capture</button>
        </div>

        <div v-else class="image-container">
          <img src="../assets/no-image.png" alt="input is missing" />
          <span>No image available</span>
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
.no-cameras-text {
  margin: 1.5rem 0;
  color: white;
}
.image-title-container {
  display: flex;
  justify-content: space-between;
}
.int-ext-container {
  margin-top: 2rem;
  display: flex;
  color: white;
  font-size: 1.5rem;
  gap: 1.5rem;
}

.int-ext-text {
  cursor: pointer;
}

.available-camera {
  background: #003b3f;
  border: 2px solid #0d6362;
  color: #115c62;
  border-radius: 8px;
  padding: 0.7rem 2rem;
  padding-bottom: 0.6rem;
  font-size: 20px;
  font-weight: 700;
  cursor: pointer;
}
.available-camera.selected {
  background: #0098a3;
  border: 2px solid #3bc7d6;
  color: #08dee9;
}
.available-cameras-container {
  margin-top: 2rem;
}
.available-cameras-text {
  color: white;
}

.camera-list {
  height: 50px;
  margin-top: 1rem;
  display: flex;
  gap: 1rem;
}

.images-container {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
}

.container {
  display: flex;
  gap: 13rem;
  margin-top: 85px;
  color: white;
  justify-content: center;
}

.camera-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  justify-content: center;
}

.text-bold {
  font-size: 26px;
  font-weight: 700;
}

.image-container {
  position: relative;
  border: 2px solid #44a9b2;
  border-radius: 8px;
  width: 640px;
  aspect-ratio: 16 / 9;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 2rem;
}

.input-image {
  width: 100%;
  height: 100%;
  border-radius: 5px;
  object-fit: cover;
}

@media (max-width: 1350px) {
  .images-container {
    grid-template-columns: repeat(1, 1fr);
  }
}

@media (max-width: 1150px) {
  .container {
    gap: 5rem;
  }
}

@media (max-width: 965px) {
  .container {
    align-items: center;
    flex-direction: column;
    gap: 3rem;
  }
  .text-bold {
    text-align: center;
  }
}
</style>
