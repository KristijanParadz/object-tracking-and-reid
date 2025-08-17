<script setup>
import { ref, onMounted, watch } from "vue";
import { socket, intrinsicLiveFeedState } from "@/socket";
import axios from "axios";
import { checkIfCameraHasIntrinsics } from "../utils/calibration";
import router from "../router/index";
import { useToast } from "vue-toast-notification";

const cameras = ref([]);
const selectedCamera = ref(null);
const imagesPreview = ref(null);
const isProcessRunning = ref(false);

const toast = useToast();

async function fetchAvailableCameras() {
  const response = await axios.get(
    `${import.meta.env.VITE_API_BASE_URL}/api/available-cameras`
  );
  cameras.value = response.data.map((camIndex) => {
    return {
      index: camIndex,
      name: `Camera ${camIndex}`,
      isCalibrated: checkIfCameraHasIntrinsics(camIndex),
    };
  });
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
  try {
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
    toast.success(`${selectedCamera.value.name} successfuly calibrated!`);
    restartProcess();
  } catch (e) {
    toast.error("Something went wrong");
  }
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
  selectedCamera.value = null;
}

onMounted(() => {
  fetchAvailableCameras();
});
</script>

<template>
  <main>
    <img src="../assets/protostar-logo.png" alt="protostar-logo" />

    <div class="int-ext-container">
      <div class="intrinsic-text">Intrinsic</div>
      <div
        @click="() => router.push('/calibration/extrinsic')"
        class="extrinsic-text"
      >
        Extrinsic
      </div>
    </div>

    <div class="available-cameras-container">
      <span class="text-bold available-cameras-text">Available Cameras</span>
      <div v-if="cameras.length > 0" class="camera-list">
        <div v-for="camera in cameras">
          <div
            :class="`available-camera ${
              camera.index === selectedCamera?.index && 'selected'
            }`"
            @click="() => selectCamera(camera.index)"
          >
            {{ camera.name }}
          </div>
          <div v-if="camera.isCalibrated" class="calibrated-text">
            CALIBRATED
          </div>
        </div>

        <div class="divider"></div>

        <button
          @click="startCalibration"
          class="available-camera"
          :class="{
            'start-button': selectedCamera != null && !isProcessRunning,
          }"
          :disabled="selectedCamera == null || isProcessRunning"
        >
          START
        </button>
      </div>

      <div v-else class="no-cameras-text">
        No cameras are currently available. Please connect your cameras.
      </div>
    </div>

    <div class="container">
      <div class="camera-container">
        <div class="image-title-container">
          <div class="only-title">
            <span class="text-bold">{{
              imagesPreview ? "Preview" : "Live Feed"
            }}</span>
            <span class="frames-saved-text"
              >{{ intrinsicLiveFeedState.framesSaved || 0 }} / 10</span
            >
          </div>
          <div v-if="imagesPreview">
            <button class="try-again-confirm-button" @click="calibrateCamera">
              CONFIRM
            </button>
            <button
              class="try-again-confirm-button try-again"
              @click="restartProcess"
            >
              Try again
            </button>
          </div>
        </div>

        <div v-if="imagesPreview" class="images-container">
          <div v-for="image in imagesPreview" class="image-container">
            <img
              :src="`data:image/jpg;base64,${image}`"
              alt="input"
              class="input-image"
            />
          </div>
        </div>

        <div
          class="live-feed-container"
          v-else-if="intrinsicLiveFeedState.image"
        >
          <div class="image-container">
            <img
              :src="`data:image/jpg;base64,${intrinsicLiveFeedState.image}`"
              alt="input"
              class="input-image"
            />
          </div>
          <button
            :disabled="!intrinsicLiveFeedState.isDetected"
            :class="{
              'available-camera': true,
              'start-button': intrinsicLiveFeedState.isDetected,
            }"
            @click="captureImage"
          >
            CAPTURE
          </button>
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
.only-title {
  width: 644px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.try-again {
  margin-left: 1rem;
}

.try-again-confirm-button {
  font-size: 18px;
  font-weight: 700;
  color: white;
  background: #0099a3;
  border: 2px solid #00e0ef;
  width: 126px;
  height: 39px;
  border-radius: 8px;
  cursor: pointer;
}
.live-feed-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}
.divider {
  width: 2px;
  height: 50px;
  background-color: #066268;
}

.frames-saved-text {
  font-weight: bold;
  margin-right: 0.5rem;
  font-size: 20px;
}

.no-cameras-text {
  margin: 1.5rem 0;
  color: white;
  margin-left: 1rem;
  font-size: 18px;
}
.image-title-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.int-ext-container {
  margin-top: 2rem;
  display: flex;
  font-size: 1.5rem;
  gap: 1.5rem;
  margin-left: 1rem;
  font-weight: 700;
}

.intrinsic-text {
  color: white;
  cursor: pointer;
}

.extrinsic-text {
  color: #066268;
  cursor: pointer;
}

.available-camera {
  text-align: center;
  background: #003b3f;
  border: 2px solid #0d6362;
  color: #115c62;
  border-radius: 8px;
  width: 160px;
  box-sizing: border-box;
  padding: 0.7rem 0;
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
  margin-left: 1rem;
}

.start-button {
  background: #23b229;
  border: 2px solid #3df34f;
  color: #3df34f;
}

.calibrated-text {
  text-align: center;
  color: white;
  font-size: 10px;
  font-weight: 700;
  margin-top: 0.4rem;
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
  margin-top: 2.5rem;
  color: white;
}

.camera-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  justify-content: center;
}

.text-bold {
  font-size: 26px;
  font-weight: 700;
  margin-left: 1rem;
}

.image-container {
  position: relative;
  border: 2px solid #44a9b2;
  border-radius: 8px;
  width: 640px;
  height: 360px;
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
