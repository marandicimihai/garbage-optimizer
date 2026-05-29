//
//  ContentView.swift
//  Garbage
//
//  Created by Mihai Marandici on 29.05.2026.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(red: 0.06, green: 0.09, blue: 0.18), Color(red: 0.12, green: 0.2, blue: 0.34)], startPoint: .topLeading, endPoint: .bottomTrailing)
                .ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Garbage Classifier")
                            .font(.system(size: 34, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                        Text("Live camera feed with fully offline on-device CoreML inference.")
                            .foregroundStyle(.white.opacity(0.8))
                    }

                    LiveCameraPreview(session: camera.session)
                        .frame(height: 360)
                        .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 24, style: .continuous)
                                .stroke(.white.opacity(0.14), lineWidth: 1)
                        )
                        .overlay(alignment: .topLeading) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(camera.predictionLabel)
                                    .font(.system(size: 24, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white)
                                Text(camera.predictionConfidence)
                                    .font(.subheadline)
                                    .foregroundStyle(.white.opacity(0.8))
                            }
                            .padding(16)
                            .background(.black.opacity(0.35))
                            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                            .padding(14)
                        }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Current prediction")
                            .font(.headline)
                            .foregroundStyle(.white.opacity(0.7))
                        Text(camera.predictionLabel)
                            .font(.system(size: 28, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                        Text(camera.predictionConfidence.isEmpty ? "Confidence will appear after the first frame." : camera.predictionConfidence)
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.white.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Status")
                            .font(.headline)
                            .foregroundStyle(.white.opacity(0.7))
                        Text(camera.statusMessage)
                            .font(.subheadline)
                            .foregroundStyle(.white)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.white.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Actuator")
                            .font(.headline)
                            .foregroundStyle(.white.opacity(0.7))
                        Text(camera.actionMessage)
                            .font(.subheadline)
                            .foregroundStyle(.white)

                        Button {
                            camera.sendPendingRoute()
                        } label: {
                            Text(camera.canSendRoute ? "Route now" : "No route available")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 12)
                                .foregroundStyle(.white)
                                .background(camera.canSendRoute ? Color.orange.opacity(0.9) : Color.white.opacity(0.16))
                                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        }
                        .disabled(!camera.canSendRoute)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.white.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

                    Text("The app uses the phone camera directly, processes frames on-device, and never sends images to a server.")
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.7))
                        .padding(.horizontal, 2)
                }
                .padding(20)
            }
        }
        .onAppear {
            camera.start()
        }
        .onDisappear {
            camera.stop()
        }
    }
}

#Preview {
    ContentView()
}
