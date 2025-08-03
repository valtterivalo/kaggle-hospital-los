/**
 * @fileoverview Healthcare AI Predictor with clinical lab values and dark theme
 */

"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface ClinicalData {
  age: number
  gender: string
  facility: string
  readmissions: number
  medical_condition: string
  creatinine: number
  glucose: number
  hematocrit: number
  bun: number
  bmi: number
  pulse: number
  respiration: number
  sodium: number
  neutrophils: number
}

interface PredictionResult {
  predicted_los: number
  confidence_interval: [number, number]
  explanation: string
  shap_values: Record<string, number>
}

export default function Home() {
  const [clinicalData, setClinicalData] = useState<ClinicalData>({
    age: 65,
    gender: "F",
    facility: "A",
    readmissions: 1,
    medical_condition: "Pneumonia",
    creatinine: 1.1,
    glucose: 120,
    hematocrit: 12.0,
    bun: 18,
    bmi: 28.5,
    pulse: 80,
    respiration: 18,
    sodium: 140,
    neutrophils: 8.5
  })

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = (field: keyof ClinicalData, value: string | number) => {
    setClinicalData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('http://localhost:8000/api/predictions/single', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(clinicalData)
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      setPrediction(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-light text-neutral-200 mb-2">
            Clinical Prediction Model
          </h1>
          <p className="text-neutral-400 text-sm">
            Length of stay prediction with clinical lab values
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Demographics */}
          <Card className="bg-neutral-900 border-neutral-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-medium text-neutral-200">Demographics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">Age</Label>
                  <Input
                    type="number"
                    min="0"
                    max="120"
                    value={clinicalData.age}
                    onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Gender</Label>
                  <Select value={clinicalData.gender} onValueChange={(value) => handleInputChange('gender', value)}>
                    <SelectTrigger className="bg-neutral-800 border-neutral-700 text-neutral-100">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-neutral-800 border-neutral-700">
                      <SelectItem value="M">Male</SelectItem>
                      <SelectItem value="F">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">Facility</Label>
                  <Select value={clinicalData.facility} onValueChange={(value) => handleInputChange('facility', value)}>
                    <SelectTrigger className="bg-neutral-800 border-neutral-700 text-neutral-100">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-neutral-800 border-neutral-700">
                      <SelectItem value="A">Facility A</SelectItem>
                      <SelectItem value="B">Facility B</SelectItem>
                      <SelectItem value="C">Facility C</SelectItem>
                      <SelectItem value="D">Facility D</SelectItem>
                      <SelectItem value="E">Facility E</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Readmissions</Label>
                  <Input
                    type="number"
                    min="0"
                    max="5"
                    value={clinicalData.readmissions}
                    onChange={(e) => handleInputChange('readmissions', parseInt(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
              </div>

              <div>
                <Label className="text-neutral-300 text-sm">Condition</Label>
                <Select value={clinicalData.medical_condition} onValueChange={(value) => handleInputChange('medical_condition', value)}>
                  <SelectTrigger className="bg-neutral-800 border-neutral-700 text-neutral-100">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-neutral-800 border-neutral-700">
                    <SelectItem value="Pneumonia">Pneumonia</SelectItem>
                    <SelectItem value="Asthma">Asthma</SelectItem>
                    <SelectItem value="Diabetes">Diabetes</SelectItem>
                    <SelectItem value="Hypertension">Hypertension</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Lab Values */}
          <Card className="bg-neutral-900 border-neutral-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-medium text-neutral-200">Lab Values</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">Creatinine (mg/dL)</Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="0.2"
                    max="5.0"
                    value={clinicalData.creatinine}
                    onChange={(e) => handleInputChange('creatinine', parseFloat(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Glucose (mg/dL)</Label>
                  <Input
                    type="number"
                    min="50"
                    max="300"
                    value={clinicalData.glucose}
                    onChange={(e) => handleInputChange('glucose', parseInt(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">Hematocrit (g/dL)</Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="4"
                    max="25"
                    value={clinicalData.hematocrit}
                    onChange={(e) => handleInputChange('hematocrit', parseFloat(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">BUN (mg/dL)</Label>
                  <Input
                    type="number"
                    min="5"
                    max="100"
                    value={clinicalData.bun}
                    onChange={(e) => handleInputChange('bun', parseInt(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">Sodium (mmol/L)</Label>
                  <Input
                    type="number"
                    min="125"
                    max="155"
                    value={clinicalData.sodium}
                    onChange={(e) => handleInputChange('sodium', parseInt(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Neutrophils (K/μL)</Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="50"
                    value={clinicalData.neutrophils}
                    onChange={(e) => handleInputChange('neutrophils', parseFloat(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Vitals & Results */}
          <Card className="bg-neutral-900 border-neutral-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-medium text-neutral-200">Vitals & Prediction</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">BMI</Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="15"
                    max="50"
                    value={clinicalData.bmi}
                    onChange={(e) => handleInputChange('bmi', parseFloat(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Pulse (bpm)</Label>
                  <Input
                    type="number"
                    min="40"
                    max="150"
                    value={clinicalData.pulse}
                    onChange={(e) => handleInputChange('pulse', parseInt(e.target.value))}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
              </div>

              <div>
                <Label className="text-neutral-300 text-sm">Respiration (breaths/min)</Label>
                <Input
                  type="number"
                  min="8"
                  max="30"
                  value={clinicalData.respiration}
                  onChange={(e) => handleInputChange('respiration', parseInt(e.target.value))}
                  className="bg-neutral-800 border-neutral-700 text-neutral-100"
                />
              </div>

              <Button 
                onClick={handlePredict} 
                disabled={loading}
                className="w-full mt-6 bg-neutral-700 hover:bg-neutral-600 text-neutral-100 border-neutral-600"
              >
                {loading ? 'Processing...' : 'Predict Length of Stay'}
              </Button>

              {error && (
                <div className="p-3 bg-red-900/20 border border-red-800 rounded text-red-300 text-sm">
                  {error}
                </div>
              )}

              {prediction && (
                <div className="space-y-4 pt-4 border-t border-neutral-800">
                  <div className="text-center">
                    <div className="text-3xl font-light text-neutral-100 mb-1">
                      {prediction.predicted_los.toFixed(1)}
                    </div>
                    <div className="text-sm text-neutral-400 mb-2">days predicted stay</div>
                    <div className="text-xs text-neutral-500">
                      Range: {prediction.confidence_interval[0].toFixed(1)} - {prediction.confidence_interval[1].toFixed(1)} days
                    </div>
                  </div>

                  {prediction.shap_values && (
                    <div className="space-y-2">
                      <div className="text-sm text-neutral-300 font-medium">Key Factors</div>
                      {Object.entries(prediction.shap_values)
                        .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
                        .slice(0, 5)
                        .map(([feature, impact]) => (
                          <div key={feature} className="flex justify-between items-center text-xs">
                            <span className="text-neutral-400 truncate">{feature.replace(/_/g, ' ')}</span>
                            <span className={impact > 0 ? 'text-red-400' : 'text-green-400'}>
                              {impact > 0 ? '+' : ''}{impact.toFixed(2)}
                            </span>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="mt-8 text-center text-xs text-neutral-600">
          Clinical AI Model v2.0 • R² = 0.971 • ~10hr accuracy
        </div>
      </div>
    </div>
  )
}