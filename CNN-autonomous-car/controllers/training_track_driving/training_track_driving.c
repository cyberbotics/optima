/*
 * Copyright 1996-2021 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description:   Autonoumous vehicle controller example
 */

#include <webots/camera.h>
#include <webots/device.h>
#include <webots/display.h>
#include <webots/gps.h>
#include <webots/keyboard.h>
#include <webots/lidar.h>
#include <webots/robot.h>
#include <webots/supervisor.h>
#include <webots/vehicle/driver.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TIME_STEP 10

// Modes
#define CW 0
#define CC 1

// enabe various 'features'
bool enable_collision_avoidance = false;
bool enable_display = false;
bool has_gps = false;
bool has_camera = false;

// cameras
WbDeviceTag camera;
WbDeviceTag camera_left;
WbDeviceTag camera_right;
int camera_width = -1;
int camera_height = -1;
double camera_fov = -1.0;

// misc variables
double speed = 0.0;
double steering_angle = 0.0;

float time_stamps[500];
float vel_stamps[500];
float angle_stamps[500];

void load_stamps(char const *timeFileName, char const *velFileName,
                 char const *angleFileName) {
  FILE *timefile = fopen(timeFileName, "r");
  FILE *velfile = fopen(velFileName, "r");
  FILE *anglefile = fopen(angleFileName, "r");

  float timeReadVar = 0.0;
  float velReadVar = 0.0;
  float angleReadVar = 0.0;

  for (int i = 0; i < 484; i++) {
    fscanf(timefile, "%f", &timeReadVar);
    fscanf(velfile, "%f", &velReadVar);
    fscanf(anglefile, "%f", &angleReadVar);
    time_stamps[i] = timeReadVar;
    vel_stamps[i] = velReadVar;
    angle_stamps[i] = angleReadVar;
  }

  fclose(timefile);
  fclose(velfile);
  fclose(anglefile);

  return 0;
}

// set target speed
void set_speed(double kmh) {
  // max speed
  if (kmh > 250.0)
    kmh = 250.0;

  speed = kmh;
  double real_speed = wbu_driver_get_current_speed();

  wbu_driver_set_cruising_speed(kmh);
}

// positive: turn right, negative: turn left
void set_steering_angle(double wheel_angle) {
  // limit the difference with previous steering_angle
  if (wheel_angle - steering_angle > 0.1)
    wheel_angle = steering_angle + 0.1;
  if (wheel_angle - steering_angle < -0.1)
    wheel_angle = steering_angle - 0.1;
  steering_angle = wheel_angle;

  // limit range of the steering angle
  if (wheel_angle > 0.5)
    wheel_angle = 0.5;
  else if (wheel_angle < -0.5)
    wheel_angle = -0.5;

  wbu_driver_set_steering_angle(wheel_angle);
}

int main(int argc, char **argv) {
  wbu_driver_init();

  // check if there is a SICK and a display
  int j = 0;
  for (j = 0; j < wb_robot_get_number_of_devices(); ++j) {
    WbDeviceTag device = wb_robot_get_device_by_index(j);
    const char *name = wb_device_get_name(device);
    if (strcmp(name, "Sick LMS 291") == 0)
      enable_collision_avoidance = true;
    else if (strcmp(name, "display") == 0)
      enable_display = true;
    else if (strcmp(name, "gps") == 0)
      has_gps = true;
    else if (strcmp(name, "camera") == 0)
      has_camera = true;
  }

  // cameras
  if (has_camera) {
    camera = wb_robot_get_device("camera");
    camera_left = wb_robot_get_device("camera_left");
    camera_right = wb_robot_get_device("camera_right");
    wb_camera_enable(camera, TIME_STEP);
    wb_camera_enable(camera_left, TIME_STEP);
    wb_camera_enable(camera_right, TIME_STEP);
    camera_width = wb_camera_get_width(camera);
    camera_height = wb_camera_get_height(camera);
    camera_fov = wb_camera_get_fov(camera);
  }

  // start engine
  if (has_camera)
    set_speed(18.0); // km/h
  wbu_driver_set_hazard_flashers(true);
  wbu_driver_set_dipped_beams(true);
  wbu_driver_set_antifog_lights(true);
  wbu_driver_set_wiper_mode(SLOW);

  // choose track direction (CC = counter-clockwise | CW = clockwise)
  int mode = CW;

  char const *const CWtimeFileName = "../../scripts/CW-files/timestamps.txt";
  char const *const CWvelFileName = "../../scripts/CW-files/velstamps.txt";
  char const *const CWangleFileName = "../../scripts/CW-files/anglestamps.txt";

  char const *const CCtimeFileName = "../../scripts/CC-files/timestamps.txt";
  char const *const CCvelFileName = "../../scripts/CC-files/velstamps.txt";
  char const *const CCangleFileName = "../../scripts/CC-files/anglestamps.txt";

  WbNodeRef car_node = wb_supervisor_node_get_from_def("AUTONOMOUS_CAR");
  WbFieldRef car_trans_field =
      wb_supervisor_node_get_field(car_node, "translation");
  WbFieldRef car_rot_field = wb_supervisor_node_get_field(car_node, "rotation");

  // train track inital poses
  double CWinitialRotation[4] = {-0.01662080811297577, 0.024790612100833716,
                                 0.9995544879041551, 0.403};
  double CWinitialTranslation[3] = {250.874, -323.838, 0.412048};
  double CCinitialRotation[4] = {0.006075581501692473, 0.0040733610068066015,
                                 0.9999732471619052, -2.80453};
  double CCinitialTranslation[3] = {254.451, -318.723, 0.37101};

  // load trajectory planning data from files and open files for data storing
  FILE *steering_file;
  FILE *speed_file;
  if (mode == CW) {
    wb_supervisor_field_set_sf_rotation(car_rot_field, CWinitialRotation);
    wb_supervisor_field_set_sf_vec3f(car_trans_field, CWinitialTranslation);
    load_stamps(CWtimeFileName, CWvelFileName, CWangleFileName);
    steering_file = fopen("../../scripts/CW-files/steering.txt", "w+");
    speed_file = fopen("../../scripts/CW-files/speed.txt", "w+");
  } else if (mode == CC) {
    wb_supervisor_field_set_sf_rotation(car_rot_field, CCinitialRotation);
    wb_supervisor_field_set_sf_vec3f(car_trans_field, CCinitialTranslation);
    load_stamps(CCtimeFileName, CCvelFileName, CCangleFileName);
    steering_file = fopen("../../scripts/CC-files/steering.txt", "w+");
    speed_file = fopen("../../scripts/CC-files/speed.txt", "w+");
  }

  // main loop
  while (wbu_driver_step() != -1) {

    static int i = 0;
    static int curr_segm = 0;
    static int elapsed_time = 0;
    static int rot_sign = 1;

    // compute segment id where car is located using time file
    if (time_stamps[curr_segm] < elapsed_time) {
      curr_segm++;
    }

    // compute error in rotation between current car rotation and ref in angle
    // file
    const double *car_rotation =
        wb_supervisor_field_get_sf_rotation(car_rot_field);

    if (car_rotation[2] < 0)
      rot_sign = -1;
    else
      rot_sign = 1;

    double rot_error =
        rot_sign * car_rotation[3] + M_PI - (angle_stamps[curr_segm] + M_PI);
    int P = 1.5;

    if (rot_error < -M_PI) {
      rot_error += 2 * M_PI;
    }

    if (rot_error > M_PI) {
      rot_error -= 2 * M_PI;
    }

    // apply speed and steering to car
    set_speed(vel_stamps[curr_segm] * 3.6);
    set_steering_angle(P * rot_error);

    // log real speed and steering to txt
    double real_car_speed = wbu_driver_get_current_speed();
    fprintf(steering_file, "%f\n", steering_angle);
    fprintf(speed_file, "%f\n", real_car_speed);

    // save camera images
    const unsigned char *camera_center_image = NULL;
    const unsigned char *camera_left_image = NULL;
    const unsigned char *camera_right_image = NULL;

    camera_center_image = wb_camera_get_image(camera);
    camera_left_image = wb_camera_get_image(camera_left);
    camera_right_image = wb_camera_get_image(camera_right);
    const char img_center[64];
    const char img_left[64];
    const char img_right[64];

    if (mode == CW) {
      snprintf(img_center, sizeof(img_center),
               "../../images/CW-images-test/center_%d.png", i);
      snprintf(img_left, sizeof(img_left),
               "../../images/CW-images-test/left_%d.png", i);
      snprintf(img_right, sizeof(img_right),
               "../../images/CW-images-test/right_%d.png", i);
    } else if (mode == CC) {
      snprintf(img_center, sizeof(img_center),
               "../../images/CC-images-test/center_%d.png", i);
      snprintf(img_left, sizeof(img_left),
               "../../images/CC-images-test/left_%d.png", i);
      snprintf(img_right, sizeof(img_right),
               "../../images/CC-images-test/right_%d.png", i);
    }

    wb_camera_save_image(camera, img_center, 100);
    wb_camera_save_image(camera_left, img_left, 100);
    wb_camera_save_image(camera_right, img_right, 100);

    ++i;
    elapsed_time = i * wb_robot_get_basic_time_step();
  }

  fclose(steering_file);
  fclose(speed_file);

  wbu_driver_cleanup();

  return 0; // ignored
}
