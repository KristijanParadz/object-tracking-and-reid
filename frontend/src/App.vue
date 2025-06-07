<script setup>
import { RouterView } from "vue-router";
</script>

<template>
  <RouterView />
</template>

<style>
@font-face {
  font-family: "Artegra Sans";
  src: url("@/assets/artegra.ttf") format("truetype");
  font-weight: normal;
  font-style: normal;
}

body {
  background: #122830;
  font-family: "Artegra Sans", sans-serif;
  padding: 1rem;
}
</style>
