<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from "vue";
import { processedImagesState, socket } from "@/socket";
import { FontAwesomeIcon } from "@fortawesome/vue-fontawesome";
import {
  faCompress,
  faExpand,
  faPause,
  faPlay,
  faRotateRight,
} from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import {
  checkIfCameraHasExtrinsics,
  checkIfCameraHasIntrinsics,
} from "../utils/calibration";

const images = computed(() => processedImagesState.images);

const isPaused = ref(false);
const cameraStates = ref({});
const cameras = ref([]);

function ensureCameraState(key) {
  if (!cameraStates.value[key]) {
    cameraStates.value[key] = { hovered: false, fullscreen: false };
  }
}

function onMouseEnter(key) {
  ensureCameraState(key);
  cameraStates.value[key].hovered = true;
}

function onMouseLeave(key) {
  cameraStates.value[key].hovered = false;
}

function pauseVideo() {
  socket.emit("pause");
}

function resumeVideo() {
  socket.emit("resume");
}

function resetVideo() {
  socket.emit("reset");
}

function togglePauseResume() {
  if (!isPaused.value) {
    pauseVideo();
    isPaused.value = true;
  } else {
    resumeVideo();
    isPaused.value = false;
  }
}

function handleReset() {
  resetVideo();
  isPaused.value = false;
}

function openFullscreen(key) {
  ensureCameraState(key);
  socket.emit("fullscreen", key);
  cameraStates.value[key].fullscreen = true;
}

function closeFullscreen(key) {
  socket.emit("exit_fullscreen");
  cameraStates.value[key].fullscreen = false;
}

function onKeydownEsc(event) {
  if (event.key === "Escape") {
    for (const cameraKey in cameraStates.value) {
      if (cameraStates.value[cameraKey].fullscreen) {
        cameraStates.value[cameraKey].fullscreen = false;
      }
    }
  }
}

async function fetchAvailableCameras() {
  const response = await axios.get(
    `${import.meta.env.VITE_API_BASE_URL}/api/available-cameras`
  );

  const camerasThatAreCalibrated = response.data.filter(
    (camIndex) =>
      checkIfCameraHasIntrinsics(camIndex) &&
      checkIfCameraHasExtrinsics(camIndex)
  );

  cameras.value = camerasThatAreCalibrated.map((camIndex) => {
    return {
      index: camIndex,
      name: `Camera ${camIndex}`,
      isSelected: false,
    };
  });
}

function selectCamera(index) {
  cameras.value[index].isSelected = !cameras.value[index].isSelected;
}

function startProcess() {
  socket.emit(
    "start",
    cameras.value.filter((camera) => camera.isSelected)
  );
}

onMounted(() => {
  fetchAvailableCameras();
  window.addEventListener("keydown", onKeydownEsc);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKeydownEsc);
});
</script>

<template>
  <main>
    <img src="../assets/protostar-logo.png" alt="protostar-logo" />
    <div class="available-cameras-container">
      <span class="text-bold available-cameras-text">Available Cameras</span>
      <div v-if="cameras.length > 0" class="camera-list">
        <div
          v-for="(camera, index) in cameras"
          :class="`available-camera ${camera.isSelected && 'selected'}`"
          @click="() => selectCamera(index)"
        >
          {{ camera.name }}
        </div>
      </div>

      <div v-else class="no-cameras-text">
        No calibrated cameras found. Please visit the Camera Calibration page to
        calibrate your cameras.<br />Once calibration is complete, the cameras
        will appear here.
      </div>
      <button @click="startProcess">Start</button>
      <router-link to="/extrinsic">Camera Calibration</router-link>
    </div>

    <div class="container">
      <div class="camera-container">
        <span class="text-bold">Broadcasts</span>
        <div
          v-if="images && Object.keys(images).length > 0"
          class="images-container"
        >
          <div
            v-for="(value, key) in images"
            :key="key"
            @mouseenter="onMouseEnter(key)"
            @mouseleave="onMouseLeave(key)"
          >
            <h3>{{ key }}</h3>
            <div class="image-container">
              <img
                :src="`data:image/jpg;base64,${value}`"
                alt="input"
                class="input-image"
              />
              <transition name="fade">
                <div
                  class="controls-overlay"
                  v-show="cameraStates[key]?.hovered"
                >
                  <div class="left">
                    <button class="play-pause-btn" @click="togglePauseResume">
                      <FontAwesomeIcon v-if="isPaused" :icon="faPlay" />
                      <FontAwesomeIcon v-else :icon="faPause" />
                    </button>
                    <button class="reset-btn" @click="handleReset">
                      <FontAwesomeIcon :icon="faRotateRight" />
                    </button>
                  </div>
                  <button
                    class="full-screen-button"
                    @click="openFullscreen(key)"
                  >
                    <FontAwesomeIcon :icon="faExpand" />
                  </button>
                </div>
              </transition>
            </div>
            <transition name="fade">
              <div
                v-if="cameraStates[key]?.fullscreen"
                class="fullscreen-overlay"
                @click.self="closeFullscreen(key)"
              >
                <div class="image-container-full" @click.stop>
                  <img
                    :src="`data:image/jpg;base64,${value}`"
                    alt="fullscreen camera"
                    class="fullscreen-image"
                  />
                  <transition name="fade">
                    <div
                      class="controls-overlay"
                      v-show="cameraStates[key]?.hovered"
                    >
                      <div class="left">
                        <button
                          class="play-pause-btn"
                          @click="togglePauseResume"
                        >
                          <FontAwesomeIcon v-if="isPaused" :icon="faPlay" />
                          <FontAwesomeIcon v-else :icon="faPause" />
                        </button>
                        <button class="reset-btn" @click="handleReset">
                          <FontAwesomeIcon :icon="faRotateRight" />
                        </button>
                      </div>
                      <button
                        class="exit-full-screen-button"
                        @click="closeFullscreen(key)"
                      >
                        <FontAwesomeIcon :icon="faCompress" />
                      </button>
                    </div>
                  </transition>
                </div>
              </div>
            </transition>
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

.image-container-full {
  position: relative;
  border: 2px solid #44a9b2;
  border-radius: 8px;
  width: 1080px;
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

.controls-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  box-sizing: border-box;
  border-radius: 6px;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.8), transparent);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 1rem;
  padding-right: 0.5rem;
}

.controls-overlay .left {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.play-pause-btn,
.reset-btn,
.full-screen-button,
.exit-full-screen-button {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: white;
}

.fullscreen-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.fullscreen-image {
  width: 100%;
  height: 100%;
  border-radius: 5px;
  object-fit: cover;
  width: 1080px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.fade-enter-to,
.fade-leave-from {
  opacity: 1;
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
