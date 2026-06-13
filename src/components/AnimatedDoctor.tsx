import React from 'react';

interface AnimatedDoctorProps {
  className?: string;
}

export function AnimatedDoctor({ className = "" }: AnimatedDoctorProps) {
  return (
    <div className={`relative ${className}`}>
      <svg
        width="80"
        height="80"
        viewBox="0 0 200 200"
        className="w-full h-full"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Doctor's head */}
        <circle
          cx="100"
          cy="70"
          r="35"
          fill="#FDBCB4"
          stroke="#333"
          strokeWidth="2"
        />

        {/* Hair */}
        <path
          d="M 65 45 Q 100 25 135 45 Q 130 35 100 30 Q 70 35 65 45"
          fill="#8B4513"
          stroke="#333"
          strokeWidth="1"
        />

        {/* Eyes */}
        <circle cx="88" cy="65" r="6" fill="white" stroke="#333" strokeWidth="1" />
        <circle cx="112" cy="65" r="6" fill="white" stroke="#333" strokeWidth="1" />
        <circle cx="88" cy="65" r="3" fill="#333" />
        <circle cx="112" cy="65" r="3" fill="#333" />

        {/* Glasses */}
        <circle cx="88" cy="65" r="12" fill="none" stroke="#333" strokeWidth="2" />
        <circle cx="112" cy="65" r="12" fill="none" stroke="#333" strokeWidth="2" />
        <line x1="100" y1="65" x2="100" y2="65" stroke="#333" strokeWidth="2" />

        {/* Nose */}
        <ellipse cx="100" cy="75" rx="3" ry="5" fill="#F4A086" />

        {/* Smile */}
        <path
          d="M 90 85 Q 100 95 110 85"
          fill="none"
          stroke="#333"
          strokeWidth="2"
          strokeLinecap="round"
        />

        {/* Body/Lab coat */}
        <rect
          x="70"
          y="100"
          width="60"
          height="80"
          fill="white"
          stroke="#333"
          strokeWidth="2"
          rx="5"
        />

        {/* Lab coat lapels */}
        <path
          d="M 70 100 L 85 120 L 70 140"
          fill="white"
          stroke="#333"
          strokeWidth="2"
        />
        <path
          d="M 130 100 L 115 120 L 130 140"
          fill="white"
          stroke="#333"
          strokeWidth="2"
        />

        {/* Stethoscope */}
        <path
          d="M 85 110 Q 100 105 115 110"
          fill="none"
          stroke="#333"
          strokeWidth="3"
          strokeLinecap="round"
        />
        <circle cx="100" cy="130" r="8" fill="silver" stroke="#333" strokeWidth="2" />

        {/* Left arm (static) */}
        <ellipse
          cx="55"
          cy="125"
          rx="15"
          ry="35"
          fill="#FDBCB4"
          stroke="#333"
          strokeWidth="2"
        />

        {/* Right arm (waving) */}
        <g className="animate-doctor-wave">
          <ellipse
            cx="145"
            cy="125"
            rx="15"
            ry="35"
            fill="#FDBCB4"
            stroke="#333"
            strokeWidth="2"
          />
          <circle
            cx="145"
            cy="95"
            r="12"
            fill="#FDBCB4"
            stroke="#333"
            strokeWidth="2"
          />
        </g>

        {/* Medical cross on coat */}
        <g fill="#3b82f6">
          <rect x="95" y="145" width="10" height="25" />
          <rect x="87" y="153" width="26" height="10" />
        </g>
      </svg>
    </div>
  );
}