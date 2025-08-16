<script setup>
import { ref, onMounted, watch, computed } from "vue";
import { socket, extrinsicLiveFeedState } from "@/socket";
import router from "../router/index";
import axios from "axios";
import {
  checkIfCameraHasExtrinsics,
  checkIfCameraHasIntrinsics,
} from "../utils/calibration";
import { useToast } from "vue-toast-notification";

const cameras = ref([]);
const imagesPreview = ref(null);
const isProcessRunning = ref(false);

const toast = useToast();

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

function getCameraName(key) {
  return cameras.value.find((cam) => key === `camera${cam.index}`).name;
}

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
  try {
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

    localStorage.setItem(
      "calibrationData",
      JSON.stringify(updatedData, null, 2)
    );
    restartProcess();
    toast.success("Cameras calibrated successfuly!");
  } catch (e) {
    toast.error("Something went wrong");
  }
}

function captureImage() {
  socket.emit("extrinsic-request-frame-save", {
    frame_number: extrinsicLiveFeedState.frameNumber,
  });
}

function restartProcess() {
  window.location.reload();
}

function formatCameraLabel(str) {
  if (str.length <= 6) return str.charAt(0).toUpperCase() + str.slice(1);
  return str.charAt(0).toUpperCase() + str.slice(1, 6) + " " + str.slice(6);
}

onMounted(() => {
  fetchCamerasThatHaveIntrinsics();
});
</script>

<template>
  <main>
    <img src="../assets/protostar-logo.png" alt="protostar-logo" />

    <div class="int-ext-container">
      <div
        class="intrinsic-text"
        @click="() => router.push('/calibration/intrinsic')"
      >
        Intrinsic
      </div>
      <div class="extrinsic-text">Extrinsic</div>
    </div>

    <div class="available-cameras-container">
      <span class="available-cameras-text"
        ><span class="only-available-cameras-text">Available Cameras</span> -
        Please select at least 2 cameras to start calibration</span
      >
      <div v-if="cameras.length > 0" class="camera-list">
        <div v-for="(camera, index) in cameras">
          <div
            :class="`available-camera ${camera.isSelected && 'selected'}`"
            @click="() => selectCamera(index)"
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
          :disabled="selectedCameras.length < 2 || isProcessRunning"
          class="available-camera"
          :class="{
            'start-button': selectedCameras.length >= 2 && !isProcessRunning,
          }"
        >
          START
        </button>
      </div>

      <div v-else class="no-cameras-text">
        No cameras are currently available. Please perform intrinsic calibration
        first.
      </div>
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
            <div class="camera-title-container">
              <div class="camera-preview-text">{{ getCameraName(key) }}</div>

              <div class="buttons-container">
                <button
                  @click="calibrateCamera"
                  class="try-again-confirm-button"
                >
                  CONFIRM
                </button>
                <button
                  @click="restartProcess"
                  class="try-again-confirm-button try-again"
                >
                  Try again
                </button>
              </div>
            </div>

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
        </div>

        <div
          v-else-if="images && Object.keys(images).length > 0"
          class="images-container"
        >
          <div v-for="(value, key) in images" :key="key">
            <div class="image-title-container">
              <span class="camera-name-text">{{ formatCameraLabel(key) }}</span>
              <span class="frames-saved-text"
                >{{ extrinsicLiveFeedState.framesSaved || 0 }} / 10</span
              >
            </div>

            <div class="live-feed-container">
              <div class="image-container">
                <img
                  :src="`data:image/jpg;base64,${value}`"
                  alt="input"
                  class="input-image"
                />
              </div>

              <button
                @click="captureImage"
                class="available-camera start-button"
              >
                CAPTURE
              </button>
            </div>
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
.camera-title-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  margin-bottom: 1rem;
}
.live-feed-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}
.camera-name-text {
  margin-left: 1rem;
}
.frames-saved-text {
  margin-right: 0.5rem;
}
.only-available-cameras-text {
  font-weight: 800;
  margin-left: 1rem;
}
.divider {
  width: 2px;
  height: 50px;
  background-color: #066268;
}

.no-cameras-text {
  margin: 1.5rem 0;
  color: white;
  margin-left: 1rem;
  font-size: 18px;
}
.single-images-preview-container {
  margin-top: 2rem;
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
  font-size: 1.5rem;
  gap: 1.5rem;
  margin-left: 1rem;
  font-weight: 700;
}

.calibrated-text {
  text-align: center;
  color: white;
  font-size: 10px;
  font-weight: 700;
  margin-top: 0.4rem;
}

.intrinsic-text {
  color: #066268;
  cursor: pointer;
}

.extrinsic-text {
  color: white;
  cursor: pointer;
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
  font-weight: 500;
  font-size: 26px;
}

.start-button {
  background: #23b229;
  border: 2px solid #3df34f;
  color: #3df34f;
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
  color: white;
  margin-top: 2.5rem;
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
