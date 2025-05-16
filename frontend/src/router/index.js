import { createRouter, createWebHistory } from "vue-router";
import Home from "../views/Home.vue";
import Calibration from "../views/Calibration.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: Home,
    },
    {
      path: "/calibration",
      name: "calibration",
      component: Calibration,
    },
  ],
});

export default router;
