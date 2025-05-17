<script setup>
import { ref, onMounted } from "vue";
import { socket, intrinsicLiveFeedState } from "@/socket";
import axios from "axios";

const cameras = ref([]);
const selectedCamera = ref(null);

async function fetchAvailableCameras() {
  const response = await axios.get(
    `${import.meta.env.VITE_API_BASE_URL}/api/available-cameras`
  );
  cameras.value = response.data.map((camIndex) => {
    return {
      index: camIndex,
      name: `Camera ${camIndex}`,
    };
  });
}

function selectCamera(index) {
  selectedCamera.value = cameras.value.find((camera) => camera.index === index);
}

function startCalibration() {
  socket.emit("start-intrinsic-calibration", {
    camera_index: selectedCamera.value.index,
  });
}

function captureImage() {
  socket.emit("intrinsic-request-frame-save", {
    frame_number: intrinsicLiveFeedState.frameNumber,
  });
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
      <div class="camera-list">
        <div
          v-for="camera in cameras"
          :class="`available-camera ${
            camera.index === selectedCamera?.index && 'selected'
          }`"
          @click="() => selectCamera(camera.index)"
        >
          {{ camera.name }}
        </div>
      </div>

      <button @click="startCalibration" :disabled="selectedCamera == null">
        Start Calibration
      </button>
    </div>

    <div class="container">
      <div class="camera-container">
        <div class="image-title-container">
          <span class="text-bold">Live Feed</span>
          <span>{{ intrinsicLiveFeedState.framesSaved || 0 }} / 10</span>
        </div>

        <div v-if="intrinsicLiveFeedState.image">
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
</style>
