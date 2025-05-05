import axios from 'axios';

const API_URL = 'http://localhost:8000'; // Replace with your backend URL if not local

const poseService = axios.create({
  baseURL: API_URL,
  timeout: 10000,
});

export const processFrame = async (base64Image, exercise = 'UNKNOWN') => {
  try {
    const response = await poseService.post('/process_frame', {
      image: base64Image.split(',')[1], // Remove "data:image/jpeg;base64," prefix
      exercise,
    });
    return response.data;
  } catch (error) {
    console.error('Error processing frame:', error);
    throw error;
  }
};

export const startRecording = async (exercise = 'UNKNOWN') => {
  try {
    const response = await poseService.post('/recording', {
      action: 'start',
      exercise,
    });
    return response.data;
  } catch (error) {
    console.error('Error starting recording:', error);
    throw error;
  }
};

export const stopRecording = async (exercise = 'UNKNOWN') => {
  try {
    const response = await poseService.post('/recording', {
      action: 'stop',
      exercise,
    });
    return response.data;
  } catch (error) {
    console.error('Error stopping recording:', error);
    throw error;
  }
};

export const setExercise = async (exercise) => {
  try {
    const response = await poseService.get(`/set_exercise/${exercise}`);
    return response.data;
  } catch (error) {
    console.error('Error setting exercise:', error);
    throw error;
  }
};

export const getRecordings = async () => {
  try {
    const response = await poseService.get('/recordings');
    return response.data;
  } catch (error) {
    console.error('Error fetching recordings:', error);
    throw error;
  }
};