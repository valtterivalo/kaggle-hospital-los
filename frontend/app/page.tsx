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
  // Demographics
  gender: string  // M or F
  facid: number   // Facility ID (1-5, maps to A-E)
  rcount: number  // Readmissions within 180 days
  
  // Clinical Conditions (binary flags 0/1)
  dialysisrenalendstage: number
  asthma: number
  irondef: number
  pneum: number
  substancedependence: number
  psychologicaldisordermajor: number
  depress: number
  psychother: number
  fibrosisandother: number
  malnutrition: number
  hemo: number
  secondarydiagnosisnonicd9: number
  
  // Lab Values (continuous)
  hematocrit: number      // g/dL
  neutrophils: number     // cells/microL  
  sodium: number          // mmol/L
  glucose: number         // mmol/L (note: dataset says mmol/L for glucose)
  bloodureanitro: number  // mg/dL (BUN)
  creatinine: number      // mg/dL
  
  // Vitals (continuous)
  bmi: number        // kg/m²
  pulse: number      // beats/min
  respiration: number // breaths/min
}

interface PredictionResult {
  predicted_los: number
  confidence_interval: [number, number]
  explanation: string
  shap_values: Record<string, number>
}

export default function Home() {
  const [clinicalData, setClinicalData] = useState<ClinicalData>({
    // Demographics
    gender: "F",
    facid: 1, // Facility A
    rcount: 1,
    
    // Clinical Conditions (0 = No, 1 = Yes)
    dialysisrenalendstage: 0,
    asthma: 0,
    irondef: 0,
    pneum: 1, // Default case: pneumonia patient
    substancedependence: 0,
    psychologicaldisordermajor: 0,
    depress: 0,
    psychother: 0,
    fibrosisandother: 0,
    malnutrition: 0,
    hemo: 0,
    secondarydiagnosisnonicd9: 2, // Typical value from dataset
    
    // Lab Values
    hematocrit: 12.0,    // g/dL
    neutrophils: 8.5,    // cells/microL
    sodium: 140,         // mmol/L
    glucose: 6.7,        // mmol/L (120 mg/dL = ~6.7 mmol/L)
    bloodureanitro: 18,  // mg/dL (BUN)  
    creatinine: 1.1,     // mg/dL
    
    // Vitals
    bmi: 28.5,
    pulse: 80,
    respiration: 18
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

  const handleNumberInputChange = (field: keyof ClinicalData, value: string, isFloat: boolean = false) => {
    const numValue = isFloat ? parseFloat(value) : parseInt(value)
    const validValue = isNaN(numValue) ? 0 : numValue
    handleInputChange(field, validValue)
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

        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          {/* Demographics & Basic Info */}
          <Card className="bg-neutral-900 border-neutral-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-medium text-neutral-200">Demographics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
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
                <div>
                  <Label className="text-neutral-300 text-sm">Facility</Label>
                  <Select value={clinicalData.facid.toString()} onValueChange={(value) => handleInputChange('facid', parseInt(value))}>
                    <SelectTrigger className="bg-neutral-800 border-neutral-700 text-neutral-100">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-neutral-800 border-neutral-700">
                      <SelectItem value="1">Facility A</SelectItem>
                      <SelectItem value="2">Facility B</SelectItem>
                      <SelectItem value="3">Facility C</SelectItem>
                      <SelectItem value="4">Facility D</SelectItem>
                      <SelectItem value="5">Facility E</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">Readmissions (180d)</Label>
                  <Input
                    type="number"
                    min="0"
                    max="5"
                    value={clinicalData.rcount}
                    onChange={(e) => handleNumberInputChange('rcount', e.target.value)}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Secondary Diagnoses</Label>
                  <Input
                    type="number"
                    min="0"
                    max="10"
                    value={clinicalData.secondarydiagnosisnonicd9}
                    onChange={(e) => handleNumberInputChange('secondarydiagnosisnonicd9', e.target.value)}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Clinical Conditions */}
          <Card className="bg-neutral-900 border-neutral-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-medium text-neutral-200">Clinical Conditions</CardTitle>
              <CardDescription className="text-neutral-500 text-xs">Check all that apply during encounter</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                {[
                  { key: 'pneum', label: 'Pneumonia' },
                  { key: 'asthma', label: 'Asthma' },
                  { key: 'dialysisrenalendstage', label: 'End-Stage Renal Disease' },
                  { key: 'depress', label: 'Depression' },
                  { key: 'psychologicaldisordermajor', label: 'Major Psych Disorder' },
                  { key: 'psychother', label: 'Other Psych Disorder' },
                  { key: 'substancedependence', label: 'Substance Dependence' },
                  { key: 'irondef', label: 'Iron Deficiency' },
                  { key: 'malnutrition', label: 'Malnutrition' },
                  { key: 'hemo', label: 'Blood Disorder' },
                  { key: 'fibrosisandother', label: 'Fibrosis/Other' }
                ].map(({ key, label }) => (
                  <div key={key} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id={key}
                      checked={clinicalData[key as keyof ClinicalData] === 1}
                      onChange={(e) => handleInputChange(key as keyof ClinicalData, e.target.checked ? 1 : 0)}
                      className="rounded border-neutral-600 bg-neutral-800 text-neutral-100"
                    />
                    <Label htmlFor={key} className="text-neutral-300 text-xs cursor-pointer">
                      {label}
                    </Label>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid lg:grid-cols-2 gap-6 mb-6">

          {/* Lab Values */}
          <Card className="bg-neutral-900 border-neutral-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-medium text-neutral-200">Laboratory Values</CardTitle>
              <CardDescription className="text-neutral-500 text-xs">Average values during encounter</CardDescription>
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
                    onChange={(e) => handleNumberInputChange('creatinine', e.target.value, true)}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">BUN (mg/dL)</Label>
                  <Input
                    type="number"
                    min="5"
                    max="100"
                    value={clinicalData.bloodureanitro}
                    onChange={(e) => handleNumberInputChange('bloodureanitro', e.target.value)}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">Glucose (mmol/L)</Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="2.0"
                    max="20.0"
                    value={clinicalData.glucose}
                    onChange={(e) => handleNumberInputChange('glucose', e.target.value, true)}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Sodium (mmol/L)</Label>
                  <Input
                    type="number"
                    min="125"
                    max="155"
                    value={clinicalData.sodium}
                    onChange={(e) => handleNumberInputChange('sodium', e.target.value)}
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
                    onChange={(e) => handleNumberInputChange('hematocrit', e.target.value, true)}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Neutrophils (cells/μL)</Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="50"
                    value={clinicalData.neutrophils}
                    onChange={(e) => handleNumberInputChange('neutrophils', e.target.value, true)}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Vitals */}
          <Card className="bg-neutral-900 border-neutral-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-medium text-neutral-200">Vital Signs</CardTitle>
              <CardDescription className="text-neutral-500 text-xs">Average values during encounter</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-neutral-300 text-sm">BMI (kg/m²)</Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="15"
                    max="50"
                    value={clinicalData.bmi}
                    onChange={(e) => handleNumberInputChange('bmi', e.target.value, true)}
                    className="bg-neutral-800 border-neutral-700 text-neutral-100"
                  />
                </div>
                <div>
                  <Label className="text-neutral-300 text-sm">Pulse (beats/min)</Label>
                  <Input
                    type="number"
                    min="40"
                    max="150"
                    value={clinicalData.pulse}
                    onChange={(e) => handleNumberInputChange('pulse', e.target.value)}
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
                  onChange={(e) => handleNumberInputChange('respiration', e.target.value)}
                  className="bg-neutral-800 border-neutral-700 text-neutral-100"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Prediction Section */}
        <div className="max-w-2xl mx-auto">
          <Card className="bg-neutral-900 border-neutral-800">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg font-medium text-neutral-200 text-center">Prediction</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <Button 
                onClick={handlePredict} 
                disabled={loading}
                className="w-full bg-neutral-700 hover:bg-neutral-600 text-neutral-100 border-neutral-600"
              >
                {loading ? 'Processing...' : 'Predict Length of Stay'}
              </Button>

              {error && (
                <div className="p-4 bg-red-900/20 border border-red-800 rounded text-red-300 text-sm">
                  {error}
                </div>
              )}

              {prediction && (
                <div className="space-y-6">
                  <div className="text-center p-6 bg-neutral-800 rounded-lg">
                    <div className="text-4xl font-light text-neutral-100 mb-2">
                      {prediction.predicted_los.toFixed(1)}
                    </div>
                    <div className="text-lg text-neutral-400 mb-3">days predicted stay</div>
                    <div className="text-sm text-neutral-500">
                      Range: {prediction.confidence_interval[0].toFixed(1)} - {prediction.confidence_interval[1].toFixed(1)} days
                    </div>
                  </div>

                  {prediction.shap_values && (
                    <div className="bg-neutral-800 rounded-lg p-4">
                      <div className="text-sm text-neutral-300 font-medium mb-4 text-center">
                        Clinical Factors (SHAP Analysis)
                      </div>
                      <div className="space-y-3">
                        {Object.entries(prediction.shap_values)
                          .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
                          .slice(0, 8)
                          .map(([feature, impact]) => (
                            <div key={feature} className="flex justify-between items-center">
                              <span className="text-neutral-400 text-sm capitalize">
                                {feature.replace(/_/g, ' ')}
                              </span>
                              <div className="flex items-center space-x-2">
                                <span className={`font-mono text-sm ${impact > 0 ? 'text-red-400' : 'text-green-400'}`}>
                                  {impact > 0 ? '+' : ''}{impact.toFixed(2)}
                                </span>
                                <span className="text-xs text-neutral-500">
                                  {impact > 0 ? 'increases' : 'decreases'}
                                </span>
                              </div>
                            </div>
                          ))}
                      </div>
                      <div className="mt-4 pt-3 border-t border-neutral-700 text-xs text-neutral-500 text-center">
                        Positive values increase stay duration • Negative values decrease stay duration
                      </div>
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