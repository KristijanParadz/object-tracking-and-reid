import { reactive } from "vue";
import { io } from "socket.io-client";

export const processedImagesState = reactive({
  connected: false,
  images: {},
});

const URL = import.meta.env.VITE_API_BASE_URL;

export const socket = io(URL, {
  path: "/socket.io",
});

socket.on("connect", () => {
  processedImagesState.connected = true;
});

socket.on("disconnect", () => {
  processedImagesState.connected = false;
});

socket.on("processed-images", (data) => {
  processedImagesState.images[data.video_id] = data.image;
});
