import { createRouter, createWebHistory } from "vue-router";
import Home from "../views/Home.vue";
import Calibration from "../views/Calibration.vue";
import ExtrinsicCalibration from "../views/ExtrinsicCalibration.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: Home,
    },
    {
      path: "/calibration/intrinsic",
      name: "calibration",
      component: Calibration,
    },
    {
      path: "/calibration/extrinsic",
      name: "extrinsic",
      component: ExtrinsicCalibration,
    },
  ],
});

export default router;
