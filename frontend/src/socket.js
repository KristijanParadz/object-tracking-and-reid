import { reactive } from "vue";
import { io } from "socket.io-client";

export const processedImagesState = reactive({
  images: {},
});

export const intrinsicLiveFeedState = reactive({
  image: null,
  frameNumber: null,
  framesSaved: null,
});

export const extrinsicLiveFeedState = reactive({
  images: {},
  frameNumber: null,
  framesSaved: null,
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

socket.on("live-feed-intrinsic", (data) => {
  intrinsicLiveFeedState.image = data.image;
  intrinsicLiveFeedState.frameNumber = data.frame_number;
  intrinsicLiveFeedState.framesSaved = data.frames_saved;
});

socket.on("live-feed-extrinsic", (data) => {
  extrinsicLiveFeedState.images = data.images;
  extrinsicLiveFeedState.frameNumber = data.frame_number;
  extrinsicLiveFeedState.framesSaved = data.frames_saved;
});
