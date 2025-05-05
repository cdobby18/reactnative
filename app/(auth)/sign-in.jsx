import { Text, View, ScrollView } from 'react-native';
import React, { useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import FormField from '../../components/FormField';
import CustomButton from '../../components/CustomButton';
import { Link } from 'expo-router';


// TO DO:
// 1. CREATE A DATABASE CONNECTION FOR AUTHENTICATION
// 2. USE THE AUTHENTICATION API TO SIGN IN THE USER


const SignIn = () => {
  const [form, setForm] = useState({
    email: '',
    password: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const submit = () => {};

  return (
    <SafeAreaView className="bg-black h-full">
      <ScrollView>
        <View className="w-full justify-center min-h-[85vh] px-4 my-6">
          <Text className="text-2xl text-[#FF3A4A] text-semibold mt-10 font-psemibold">
            Login to Powerlift
          </Text>
      
          <FormField
            title="Email"
            value={form.email}
            handleChangeText={(e) => setForm((prevState) => ({
              ...prevState,
              email: e
            }))}
            otherStyles="mt-7"
            keyboardType="email-address"
          />

          <FormField
            title="Password"
            value={form.password}
            handleChangeText={(e) => setForm((prevState) => ({
              ...prevState,
              password: e
            }))}
            otherStyles="mt-7"
          />

          <CustomButton
            title="Sign In"
            handlePress={submit}
            containerStyles="mt-7"
            isLoading={isSubmitting}
          />

          <View className="justify-center pt-5 flex-row gap-2">
            <Text className="text-lg text-gray-100 font-pregular">
              Register an Account:
            </Text>
            <Link href="/sign-up" className="text-lg font-psemibold text-[#FF3A4A]">
              Sign Up
            </Link>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default SignIn;
