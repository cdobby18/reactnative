import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import CustomButton from '../../components/CustomButton';

// Base URL for the backend
const BASE_URL = 'http://192.168.1.197:8000'; // Replace with your actual IP


const Feedback = ({ onUploadAnother, feedback, score, exercise }) => (
  <View className="my-2 px-2 space-y-2">
    <Text className="text-2xl text-[#FF3A4A] font-semibold">
      SCORE: Your score is {score || 'N/A'}, {score < 85 ? 'Needs Improvement' : 'Good Form'}.
    </Text>
    <View className="w-full h-1 bg-white my-4" />
    <Text className="text-xl font-bold text-[#FF3A4A] text-center">RADAR CHART</Text>
    <View className="bg-white/20 p-5 rounded-lg items-center w-full self-center">
      <Image source={require('../../assets/images/radarchart.png')} className="w-64 h-64" />
    </View>
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
  const [hasPermission, setHasPermission] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [score, setScore] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      console.log('📸 Camera Permission Status:', status);  // Check if permissions are granted
      setHasPermission(status === 'granted');
      
      if (status !== 'granted') {
        Alert.alert('Camera Permission Denied', 'Please allow camera access in settings.');
      }
    })();
  }, []);

  const setExercise = async (exercise) => {
    try {
      await axios.get(`${BASE_URL}/set_exercise/${exercise}`);
      console.log(`Exercise set to ${exercise}`);
      setSelectedExercise(exercise);
    } catch (error) {
      console.error('Error setting exercise:', error);
      Alert.alert('Error', 'Failed to set exercise type.');
    }
  };

  const toggleRecording = async () => {
    if (!selectedExercise) {
      Alert.alert('Error', 'Please select an exercise first.');
      return;
    }
  
    try {
      const action = isRecording ? 'stop' : 'start';
      const response = await axios.post(`${BASE_URL}/recording`, {
        action,
        exercise: selectedExercise,
      });
  
      console.log(`${action} recording response:`, response.data);
  
      if (action === 'start') {
        setIsRecording(true);
      } else {
        setIsRecording(false);
        setShowFeedback(true);
        
        // After recording stops, fetch feedback from /pose_data
        const feedbackResponse = await axios.get(`${BASE_URL}/pose_data`);
        console.log('Feedback Response:', feedbackResponse.data); // You can log the response to check the data format
  
        setFeedback(feedbackResponse.data.feedback);  // Set the feedback in the state
        setScore(feedbackResponse.data.score); // Assuming the backend sends a 'score' field
      }
    } catch (error) {
      console.error('Error toggling recording:', error);
      Alert.alert('Error', 'Failed to control recording.');
    }
  };
  

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