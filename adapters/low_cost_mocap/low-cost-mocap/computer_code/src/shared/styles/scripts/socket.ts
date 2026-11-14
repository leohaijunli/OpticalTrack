import { io } from 'socket.io-client';

export const API_URL = import.meta.env.VITE_MOCAP_API_URL ||
  `http://${window.location.hostname || "localhost"}:3001`;
export const socket = io(API_URL);
