import React, { useState, useEffect, useRef } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, Alert, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera } from 'expo-camera';
import axios from 'axios';
import CustomButton from '../../components/CustomButton';

const windowHeight = Dimensions.get('window').height;

const Feedback = ({ onUploadAnother, feedback, score, exercise }) => (
  <View className="my-2 px-2 space-y-2">
    <Text className="text-2xl text-[#FF3A4A] font-semibold">
      SCORE: Your score is {score || 'N/A'}, {score < 85 ? 'Needs Improvement' : 'Good Form'}.
    </Text>
    <View className="w-full h-1 bg-white my-4" />
    <Text className="text-xl font-bold text-[#FF3A4A] text-center mt-4">VIDEO ANALYSIS FEEDBACK</Text>
    <View className="bg-white/20 p-5 rounded-lg items-center w-full self-center">
      <Image source={require('../../assets/images/feedback.png')} className="w-72 h-40 mb-2" />
      <Text className="text-white text-lg font-bold">{exercise}</Text>
      <Text className="text-white text-base mt-2">{feedback || 'No feedback available'}</Text>
    </View>
    <CustomButton
      title="RECORD ANOTHER VIDEO"
      handlePress={onUploadAnother}
      containerStyles="mt-6 w-72 items-center self-center"
    />
  </View>
);


const Analysis = () => {
  const [selectedExercise, setSelectedExercise] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [hasPermission, setHasPermission] = useState(null);
  const [feedback, setFeedback] = useState('');
  const [score, setScore] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const cameraRef = useRef(null);

  useEffect(() => {
    console.log('Camera Module:', Camera); // Debug: Check if Camera is imported
    console.log('Camera Constants:', Camera?.Constants); // Debug: Check if Constants exists
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      console.log('📸 Camera Permission Status:', status);
      setHasPermission(status === 'granted');
      if (status !== 'granted') {
        Alert.alert('Camera Permission Denied', 'Please allow camera access in settings.');
      }
    })();
  }, []);

  const setExercise = async (exercise) => {
    try {
      const response = await axios.get(`${BASE_URL}/set_exercise/${exercise}`);
      console.log(`Exercise set to ${exercise}`, response.data);
      setSelectedExercise(exercise);
    } catch (error) {
      console.error('Error setting exercise:', error.message);
      Alert.alert('Error', 'Failed to set exercise type. Is the backend running?');
    }
  };

  const captureFrame = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({ base64: true });
        return photo.base64;
      } catch (error) {
        console.error('Error capturing frame:', error.message);
        return null;
      }
    }
    return null;
  };

  const toggleRecording = async () => {
    if (!selectedExercise) {
      Alert.alert('Error', 'Please select an exercise first.');
      return;
    }

    try {
      if (!isRecording) {
        setIsRecording(true);
        const response = await axios.post(`${BASE_URL}/recording`, {
          action: 'start',
          exercise: selectedExercise,
        });
        console.log('Started recording:', response.data);
      } else {
        setIsRecording(false);
        const response = await axios.post(`${BASE_URL}/recording`, {
          action: 'stop',
          exercise: selectedExercise,
        });
        console.log('Stopped recording:', response.data);

        const frame = await captureFrame();
        if (frame) {
          const frameResponse = await axios.post(`${BASE_URL}/process_frame`, {
            image: frame,
            exercise: selectedExercise,
          });
          setFeedback(frameResponse.data.feedback || 'No feedback available');
          setScore(frameResponse.data.score || 80);
        } else {
          console.error('No frame captured for analysis');
          setFeedback('No frame captured for analysis');
          setScore(0);
        }

        setShowFeedback(true);
      }
    } catch (error) {
      console.error('Error toggling recording:', error.message);
      Alert.alert('Error', 'Failed to control recording. Check if backend is running and reachable.');
      setIsRecording(false);
    }
  };

  if (hasPermission === null) {
    return (
      <SafeAreaView className="bg-black h-full">
        <View className="flex-1 justify-center items-center">
          <Text className="text-white">Requesting camera permissions...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (hasPermission === false) {
    return (
      <SafeAreaView className="bg-black h-full">
        <View className="flex-1 justify-center items-center">
          <Text className="text-white">No access to camera. Please enable in settings.</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView className="bg-black h-full">
      <ScrollView contentContainerStyle={{ paddingBottom: 20 }}>
        <View className="my-10 px-4 space-y-8">
          <View className="justify-start items-start mb-6 mt-10">
            <Text className="text-3xl font-psemibold text-gray-100 mb-2">
              {showFeedback ? 'Feedback' : 'Record Exercise Video'}
            </Text>
            <Text className="text-2xl text-[#FF3A4A] font-psemibold">Guest</Text>
          </View>

          {!showFeedback ? (
            <>
              <Text className="text-xl text-[#FF3A4A] font-bold text-center">VIDEO RECORD</Text>
              {hasPermission && selectedExercise && Camera?.Constants ? (
                <View style={{ width: '100%', height: windowHeight * 0.7, aspectRatio: 16 / 9 }}>
                  <Camera
                    ref={cameraRef}
                    style={{ width: '100%', height: '100%' }}
                    type={Camera.Constants.Type.back}
                  />
                </View>
              ) : (
                <View style={{ width: '100%', height: windowHeight * 0.7, backgroundColor: '#333', justifyContent: 'center', alignItems: 'center' }}>
                  <Text className="text-white">
                    {selectedExercise ? 'Camera not available. Ensure camera module is loaded.' : 'Select an exercise to start recording.'}
                  </Text>
                </View>
              )}
              <View className="items-center space-y-6">
                <View className="flex-row space-x-8 mt-5">
                  {[
                    { name: 'DEADLIFT', icon: require('../../assets/icons/deadlift.png') },
                    { name: 'BENCH', icon: require('../../assets/icons/bench.png') },
                  ].map((exercise, index) => (
                    <TouchableOpacity
                      key={index}
                      className="items-center"
                      onPress={() => setExercise(exercise.name)}
                    >
                      <Image source={exercise.icon} className="w-32 h-32 mb-2" />
                      <Text className="text-[#FF3A4A] font-bold text-center">
                        {exercise.name} VIDEO RECORD
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                <TouchableOpacity
                  className="items-center"
                  onPress={() => setExercise('SQUAT')}
                >
                  <Image source={require('../../assets/icons/squat.png')} className="w-32 h-32 mb-2" />
                  <Text className="text-[#FF3A4A] font-bold text-center">SQUAT VIDEO RECORD</Text>
                </TouchableOpacity>
              </View>
              {selectedExercise && (
                <View className="items-center space-y-4 mt-6">
                  <CustomButton
                    title={isRecording ? `STOP RECORDING ${selectedExercise}` : `RECORD ${selectedExercise}`}
                    handlePress={toggleRecording}
                    containerStyles="mt-6 w-72 items-center"
                  />
                </View>
              )}
            </>
          ) : (
            <Feedback
              onUploadAnother={() => {
                setShowFeedback(false);
                setSelectedExercise(null);
                setFeedback('');
                setScore(null);
                setIsRecording(false);
              }}
              feedback={feedback}
              score={score}
              exercise={selectedExercise}
            />
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default Analysis;
