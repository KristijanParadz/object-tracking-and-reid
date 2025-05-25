<script setup>
import { ref, onMounted, watch, computed } from "vue";
import { socket, extrinsicLiveFeedState } from "@/socket";
import axios from "axios";
import {
  checkIfCameraHasExtrinsics,
  checkIfCameraHasIntrinsics,
} from "../utils/calibration";

const cameras = ref([]);
const imagesPreview = ref(null);
const isProcessRunning = ref(false);

const images = computed(() => {
  return extrinsicLiveFeedState.images;
});
const selectedCameras = computed(() => {
  return cameras.value.filter((camera) => camera.isSelected);
});

async function fetchCamerasThatHaveIntrinsics() {
  const response = await axios.get(
    `${import.meta.env.VITE_API_BASE_URL}/api/available-cameras`
  );

  const camerasThatHaveIntrinsics = response.data.filter((camIndex) =>
    checkIfCameraHasIntrinsics(camIndex)
  );

  cameras.value = camerasThatHaveIntrinsics.map((camIndex) => {
    return {
      index: camIndex,
      name: `Camera ${camIndex}`,
      isSelected: false,
      isCalibrated: checkIfCameraHasExtrinsics(camIndex),
    };
  });
}

watch(
  () => extrinsicLiveFeedState.framesSaved,
  () => {
    if (extrinsicLiveFeedState.framesSaved === 10) getCapturedImagesPreview();
  }
);

async function getCapturedImagesPreview() {
  const response = await axios.get(
    `${import.meta.env.VITE_API_BASE_URL}/api/extrinsic-images-preview`
  );
  imagesPreview.value = response.data;
}

function selectCamera(index) {
  cameras.value[index].isSelected = !cameras.value[index].isSelected;
}

function startCalibration() {
  isProcessRunning.value = true;
  socket.emit("start-extrinsic-calibration", {
    camera_indexes: selectedCameras.value.map((camera) => camera.index),
  });
}

async function calibrateCamera() {
  const existingData = JSON.parse(
    localStorage.getItem("calibrationData") || "[]"
  );

  const response = await axios.post(
    `${import.meta.env.VITE_API_BASE_URL}/api/extrinsic-camera-calibration`,
    {
      intrinsics: existingData,
    }
  );

  const extrinsics = response.data;

  // Add R and t to each matching camera object
  const updatedData = existingData.map((camera) => {
    const extrinsic = extrinsics[camera.index];
    if (extrinsic) {
      return {
        ...camera,
        R: extrinsic.R,
        t: extrinsic.t,
      };
    }
    return camera;
  });

  localStorage.setItem("calibrationData", JSON.stringify(updatedData, null, 2));
  restartProcess();
}

function captureImage() {
  socket.emit("extrinsic-request-frame-save", {
    frame_number: extrinsicLiveFeedState.frameNumber,
  });
}

function restartProcess() {
  window.location.reload();
}

onMounted(() => {
  fetchCamerasThatHaveIntrinsics();
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
      <span class="text-bold available-cameras-text"
        >Available Cameras - Please select at least 2 cameras to start
        calibration</span
      >
      <div v-if="cameras.length > 0" class="camera-list">
        <div
          v-for="(camera, index) in cameras"
          :class="`available-camera ${camera.isSelected && 'selected'}`"
          @click="() => selectCamera(index)"
        >
          {{ `${camera.name} ${camera.isCalibrated ? "je" : "nije"}` }}
        </div>
      </div>

      <div v-else class="no-cameras-text">
        No cameras are currently available. Please perform intrinsic calibration
        first.
      </div>

      <button
        @click="startCalibration"
        :disabled="selectedCameras.length < 2 || isProcessRunning"
      >
        Start Calibration
      </button>
    </div>

    <div class="container">
      <div class="camera-container">
        <span class="text-bold">{{
          imagesPreview ? "Preview" : "Live Feed"
        }}</span>

        <div v-if="imagesPreview && Object.keys(imagesPreview).length > 0">
          <div
            v-for="(cameraImages, key) in imagesPreview"
            :key="key"
            class="single-images-preview-container"
          >
            <div class="camera-preview-text">{{ key }}</div>
            <div class="images-container">
              <div v-for="image in cameraImages" class="image-container">
                <img
                  :src="`data:image/jpg;base64,${image}`"
                  alt="input"
                  class="input-image"
                />
              </div>
            </div>
          </div>
          <button @click="calibrateCamera">Calibrate</button>
          <button @click="restartProcess">Try again</button>
        </div>

        <div
          v-else-if="images && Object.keys(images).length > 0"
          class="images-container"
        >
          <div v-for="(value, key) in images" :key="key">
            <div class="image-title-container">
              <span>{{ key }}</span>
              <span>{{ extrinsicLiveFeedState.framesSaved || 0 }} / 10</span>
            </div>

            <div class="image-container">
              <img
                :src="`data:image/jpg;base64,${value}`"
                alt="input"
                class="input-image"
              />
            </div>

            <button @click="captureImage">Capture</button>
          </div>
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
.single-images-preview-container {
  margin-top: 2rem;
}
.camera-preview-text {
  font-size: 18px;
  margin-bottom: 1rem;
}
.image-title-container {
  display: flex;
  justify-content: space-between;
  margin-bottom: 1rem;
  font-size: 18px;
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
