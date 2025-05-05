import { View, Text, ScrollView } from 'react-native';
import React, { useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import FormField from '../../components/FormField';
import CustomButton from '../../components/CustomButton';

const Profile = () => {
  const [form, setForm] = useState({
    username: '',
    email: '',
    password: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const updateProfile = () => {
    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      console.log('Profile updated:', form);
    }, 2000);
  };

  return (
    <SafeAreaView className="bg-black h-full">
      <ScrollView contentContainerStyle={{ paddingBottom: 20 }}>
        <View className="my-6 px-4 space-y-6">
          <View className="justify-start items-start mb-6 mt-10">
            <Text className="text-3xl font-psemibold text-gray-100 mb-2">
              User Profile
            </Text>
            <Text className="text-3xl text-[#FF3A4A] font-psemibold">
              Guest
            </Text>
          </View>
          <View className="items-center">
            <Text className="text-2xl text-[#FF3A4A] font-psemibold mt-1 mb-6">
              Update Profile
            </Text>

            <FormField
              title="Username"
              value={form.username}
              handleChangeText={(e) =>
                setForm((prevState) => ({
                  ...prevState,
                  username: e
                }))
              }
              otherStyles="mt-5 w-72"
            />

            <FormField
              title="Email"
              value={form.email}
              handleChangeText={(e) =>
                setForm((prevState) => ({
                  ...prevState,
                  email: e
                }))
              }
              otherStyles="mt-5 w-72"
              keyboardType="email-address"
            />

            <FormField
              title="Password"
              value={form.password}
              handleChangeText={(e) =>
                setForm((prevState) => ({
                  ...prevState,
                  password: e
                }))
              }
              otherStyles="mt-5 w-72"
            />

            <CustomButton
              title="Update Changes"
              handlePress={updateProfile}
              containerStyles="mt-10 w-72"
              isLoading={isSubmitting}
            />
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default Profile;
