// src/components/KeywordInput.js
import React, { useState, useRef } from "react";
import '../index.css';
import SubmitButton from "./SubmitButton";
import { FormContainer, InputField, KeywordInputForm, TextElement, DownloadButton } from "./Styles/Container/KeywordInput.style";
import ProgressBarComponent from "./ProgressBarComponent";
import axios from "axios";
import VideoSourceManager from "./VideoSourceManager";

const KeywordInput = () => {
  const [keyword, setKeyword] = useState("");
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDownloadable, setIsDownloadable] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [showVideoSourceManager, setShowVideoSourceManager] = useState(false);
  const [realResultData, setRealResultData] = useState(null);
  const [aiResultData, setAiResultData] = useState(null);
  const [isButtonLoading, setButtonLoading] = useState(false);

  const startTimeRef = useRef(null);
  const timerRef = useRef(null);

  const handleInputChange = (e) => {
    setKeyword(e.target.value);
  };

  const fetchImageAsFile = async (url, index) => {
    const response = await fetch(url);
    const blob = await response.blob();
    const ext = url.split('.').pop().split(/\#|\?/)[0];
    const filename = `image_${index}.${ext}`;
    return new File([blob], filename, { type: blob.type });
  };

  const handleSubmit = async () => {
    setButtonLoading(true);

    if (keyword.trim() === "") {
      alert("키워드를 입력하세요!");
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    setProgressText("뉴스 스크랩 중...");
    setIsDownloadable(false);
    setVideoUrl(null);

    try {
      const scrapRes = await axios.post("/scrap", { query: keyword }, { timeout: 900000 });
      setProgress(25);
      setProgressText("스크립트 생성 중...");

      const scriptRes = await axios.post("/script", {
        query: keyword,
        news: scrapRes.data.news,
      }, { timeout: 900000 });
      setProgress(50);
      setProgressText("이미지 매칭 중...");

      const readlImageJob = await axios.post("/image_real", {
        query: keyword,
        script: scriptRes.data,
      });
      const aiImageJob = await axios.post("/image_ai", {
        query: keyword,
        script: scriptRes.data,
      });
      const jobId_ai = aiImageJob.data.job_id;
      const jobId_real = readlImageJob.data.job_id;

      let realResult = null;
      let aiResult = null;
      let realDone = false;
      let aiDone = false;

      startTimeRef.current = Date.now();
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = setInterval(() => {
        const now = Date.now();
        const elapsed = Math.floor((now - startTimeRef.current) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        setProgressText(`이미지 매칭 중... (작업 대기 ${minutes}분 ${seconds}초)`);
      }, 1000);

      while (!realDone || !aiDone) {
        if (!realDone) {
          const statusRes = await axios.get(`/status/${jobId_real}`);
          if (statusRes.data.status === "FAILURE") throw new Error("실제 이미지 매칭 실패");
          if (statusRes.data.status === "SUCCESS") {
            const resultRes = await axios.get(`/result/${jobId_real}`);
            realResult = resultRes.data.result;
            setRealResultData(realResult[0]);
            realDone = true;
          }
        }
        if (!aiDone) {
          const statusRes = await axios.get(`/status/${jobId_ai}`);
          if (statusRes.data.status === "FAILURE") throw new Error("AI 이미지 생성 실패");
          if (statusRes.data.status === "SUCCESS") {
            const resultRes = await axios.get(`/result/${jobId_ai}`);
            aiResult = resultRes.data.result;
            setAiResultData(aiResult[0]);
            aiDone = true;
          }
        }
        await new Promise(resolve => setTimeout(resolve, 10000));
      }

      clearInterval(timerRef.current);
      setProgress(75);
      setProgressText("이미지 매칭 완료");
      setTimeout(() => {
        setProgressText("사용자 이미지 선택 대기 중...");
        setShowVideoSourceManager(true);
      }, 2000);
    } catch (error) {
      console.error("에러 발생:", error);
      setProgressText("에러가 발생했습니다. 콘솔을 확인해주세요.");
      setButtonLoading(false);
    }
  };

  const handleVideoSubmit = async (resultData) => {
    setProgress(90);
    setProgressText("비디오 생성 요청 중...");
    setRealResultData(null);
    setAiResultData(null);
    setShowVideoSourceManager(false);

    const formData = new FormData();
    for (let i = 0; i < resultData.length; i++) {
      const { script, imageType } = resultData[i];
      const imageUrl = imageType === 0
        ? realResultData.image[i]
        : aiResultData.image[i];
      const file = await fetchImageAsFile(imageUrl, i);
      formData.append("images", file);
      formData.append("captions", script);
    }

    try {
      const res = await axios.post("/video", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const jobId = res.data.job_id;

      startTimeRef.current = Date.now();
      timerRef.current = setInterval(async () => {
        const now = Date.now();
        const elapsed = Math.floor((now - startTimeRef.current) / 1000);
        setProgressText(`비디오 생성 중... (${elapsed}초 경과)`);

        const statusRes = await axios.get(`/status/${jobId}`);
        if (statusRes.data.status === "SUCCESS") {
          clearInterval(timerRef.current);
          const resultRes = await axios.get(`/result/${jobId}`);
          setVideoUrl(resultRes.data.video_url);
          setProgress(100);
          setProgressText("완료되었습니다. 다운로드 가능!");
          setIsDownloadable(true);
          setIsProcessing(false);
          setButtonLoading(false);
        } else if (statusRes.data.status === "FAILURE") {
          clearInterval(timerRef.current);
          setProgressText("비디오 생성 실패");
          setIsProcessing(false);
          setButtonLoading(false);
        }
      }, 5000);
    } catch (err) {
      console.error(err);
      setProgressText("비디오 생성 실패");
      setIsProcessing(false);
      setButtonLoading(false);
    }
  };

  const handleDownload = () => {
    if (videoUrl) {
      const timestamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
      const safeKeyword = keyword.replace(/[^가-힣a-zA-Z0-9]/g, "_");
      const filename = `${timestamp}_${safeKeyword}.mp4`;
      const link = document.createElement("a");
      link.href = videoUrl;
      link.download = filename;
      link.click();
    }
  };

  return (
    <KeywordInputForm>
      <TextElement>최신 뉴스와 관련된 키워드를 입력하고, 요약 영상을 만들어보세요!</TextElement>
      <FormContainer>
        <InputField
          type="text"
          placeholder="키워드를 입력하세요"
          value={keyword}
          onChange={handleInputChange}
          disabled={isProcessing}
        />
        <SubmitButton onSubmit={handleSubmit} isLoading={isButtonLoading} setLoading={setButtonLoading} />
      </FormContainer>

      {isProcessing && (
        <>
          <ProgressBarComponent progress={progress} />
          <div style={{ marginTop: '10px', color: '#333' }}>{progressText}</div>
        </>
      )}

      {isDownloadable && videoUrl && (  
        <DownloadButton onClick={handleDownload}>다운로드</DownloadButton>
      )}

      {realResultData && aiResultData && showVideoSourceManager && (
        <VideoSourceManager
          realResult={realResultData}
          aiResult={aiResultData}
          onSubmit={handleVideoSubmit}
        />
      )}
    </KeywordInputForm>
  );
};

export default KeywordInput;
